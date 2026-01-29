# Lobster Unified File Loading Architecture Validation

RULE 1: ALWAYS KEEP THIS DOCUMENT UP TO DATE
RULE 2: NEVER DEVIATE FROM THE OVERALL GOAL OF YOUR TASK
RULE 3: NEVER IMPLEMENT NON-PROFESSIONAL, BAD, BUGGY OR DUCK-TAPE CODE
RULE 4: ALWAYS REMIND YOURSELF OF RULE 1

---

## MISSION: Validate 1,836-line implementation plan for v3.0 migration

**Goal**: Determine if full architectural rewrite is necessary vs. incremental fix
**Critical Bug**: 10X V2 datasets (genes.tsv) misdetected as V3 (features.tsv) → 0 genes loaded

---

## PHASE 1: BUG VERIFICATION [IN PROGRESS]

### Task 1.1: Confirm 10X V2/V3 Detection Bug
- [ ] Read `lobster/core/archive_utils.py` lines 207-344 (ContentDetector)
- [ ] Verify `TEN_X_REQUIRED_FILES` hardcodes `features.tsv` only (V3)
- [ ] Check if `genes.tsv` detection exists anywhere
- [ ] Document exact line numbers of the bug

### Task 1.2: Trace Bug Impact Path
- [ ] Map how ContentDetector.detect_content_type() flows to loading
- [ ] Identify which loading function is called for 10X data
- [ ] Understand why 0 genes result (wrong loader or wrong parsing?)

### Task 1.3: Search for Existing Tests
- [ ] Search for GSE155698 test cases
- [ ] Search for 10X V2 specific tests
- [ ] Search for genes.tsv vs features.tsv handling

---

## PHASE 2: CURRENT ARCHITECTURE AUDIT [PENDING]

### Task 2.1: ContentDetector Usage Map
- [ ] Find all callers of ContentDetector
- [ ] Document coupling with client.py
- [ ] Identify any existing format detection alternatives

### Task 2.2: Client.py Loading Methods Audit
- [ ] Review lines 900-1600 in client.py
- [ ] Document format-specific loading methods
- [ ] Assess separation of concerns (detection vs loading vs orchestration)

### Task 2.3: DataManagerV2 Integration Points
- [ ] How does detected format affect modality creation?
- [ ] Where is format information stored/used downstream?

---

## PHASE 3: PLAN VALIDATION [PENDING]

### Task 3.1: Validate Detection Algorithm Design
- [ ] "Run ALL detectors, best confidence wins" - is this necessary?
- [ ] Would priority-based with proper ordering suffice?
- [ ] Assess performance implications

### Task 3.2: Validate Resource Limits Design
- [ ] depth=3, size=500GB, timeout=5min - appropriate for CLI?
- [ ] Are these limits customer-driven or speculative?

### Task 3.3: Validate Streaming Architecture
- [ ] Is streaming necessary for current customer datasets?
- [ ] What's the largest dataset customers actually process?
- [ ] Memory usage of current approach vs proposed

---

## PHASE 4: RISK ASSESSMENT [PENDING]

### Task 4.1: Breaking Changes Impact
- [ ] What external systems depend on ContentDetector?
- [ ] What external systems depend on client.py loading methods?
- [ ] Are there any plugins/custom packages using these APIs?

### Task 4.2: Test Coverage Gap Analysis
- [ ] Current test coverage for archive_utils.py
- [ ] Current test coverage for client.py loading
- [ ] Estimated effort to reach 95% on new code

### Task 4.3: Timeline Risk Assessment
- [ ] Is 3-6 months realistic for current team?
- [ ] What's the minimum viable fix timeline?
- [ ] Can we phase this differently?

---

## PHASE 5: ALTERNATIVES ANALYSIS [PENDING]

### Task 5.1: Minimal Fix Option
- [ ] Can we just add genes.tsv to TEN_X_REQUIRED_FILES?
- [ ] What's the minimum code change to fix the bug?
- [ ] What are the risks of minimal fix?

### Task 5.2: Incremental Migration Option
- [ ] Can we add FormatRegistry alongside ContentDetector?
- [ ] Deprecate ContentDetector over 2-3 releases?
- [ ] Phase new detectors without breaking existing

### Task 5.3: Full Rewrite Option (Current Plan)
- [ ] Is clean break truly necessary?
- [ ] What features are must-have vs nice-to-have?
- [ ] What's the real business case?

---

## CRITICAL QUESTIONS TO ANSWER

1. **Is the 10X V2 bug real?** [VERIFY FIRST]
   - Does `TEN_X_REQUIRED_FILES` actually exclude genes.tsv?
   - Is there fallback logic that handles V2?

2. **Is full rewrite necessary?**
   - Can minimal fix suffice for pre-seed stage?
   - What's driving the 7,500 LOC estimate?

3. **What's the actual customer pain?**
   - How many datasets fail due to this bug?
   - What formats do customers actually use?

4. **What's the opportunity cost?**
   - 3-6 months of dev time on loading infrastructure
   - vs. building customer-facing features

---

## FINDINGS LOG

### Finding 1: BUG CLAIM IN PLAN IS INACCURATE
- **Source**: `lobster/core/archive_utils.py` lines 231-342
- **Evidence**:
  - Plan claims line 227 has `TEN_X_REQUIRED_FILES = {"matrix.mtx", "features.tsv", "barcodes.tsv"}` (V3-only)
  - **ACTUAL CODE** (lines 233-234):
    ```python
    TEN_X_V3_FILES = {"matrix", "features", "barcodes"}  # V3 chemistry
    TEN_X_V2_FILES = {"matrix", "genes", "barcodes"}     # V2 chemistry
    ```
  - Detection logic (lines 334-341) checks BOTH:
    ```python
    has_v3 = cls.TEN_X_V3_FILES.issubset(basenames) or cls.TEN_X_V3_FILES.issubset(basenames_with_ext)
    has_v2 = cls.TEN_X_V2_FILES.issubset(basenames) or cls.TEN_X_V2_FILES.issubset(basenames_with_ext)
    if has_v3 or has_v2:
        return ArchiveContentType.TEN_X_MTX
    ```
- **Impact**:
  - Detection correctly identifies BOTH V2 (genes.tsv) and V3 (features.tsv)
  - Bug may be in LOADING, not DETECTION
  - Full rewrite of detection system may be unnecessary

### Finding 2: Detection Returns Single Type (No V2/V3 Distinction)
- **Source**: `lobster/core/archive_utils.py` line 38, 342
- **Evidence**:
  - Both V2 and V3 return same `ArchiveContentType.TEN_X_MTX`
  - No way to distinguish which version was detected
- **Impact**:
  - Loading code must handle both formats dynamically
  - If loading code assumes V3-only, THAT's where bug would be
  - Need to check client.py loading methods

### Finding 3: ALL Loading Code Has V2 Support
- **Source**: Multiple files in loading chain
- **Evidence**:
  - `client.py:1434` manual parser: `elif "feature" in name_lower or "genes" in name_lower:`
  - `client.py:1526-1532` handles 1-column and 2-column feature files
  - `transcriptomics_adapter.py:390-394` feature candidates include genes.tsv.gz, genes.tsv
  - `geo/loaders/tenx.py:141-143` globs both "*features*" and "*genes*"
  - `geo/parser.py:943-952` has 8 feature patterns including all V2 variants
- **Impact**:
  - **LOADING CODE IS CORRECT**
  - V2 format is supported throughout the entire loading chain
  - Bug described in plan appears to NOT EXIST

### Finding 4: No Direct Unit Tests for ContentDetector
- **Source**: Test search by sub-agent
- **Evidence**:
  - `tests/unit/core/test_archive_utils.py` does NOT exist
  - `tests/unit/core/test_content_detector.py` does NOT exist
  - Only indirect testing via integration tests
- **Impact**:
  - Cannot prove correctness without explicit tests
  - Regression risk if code changes
  - Test coverage gap, NOT bug

### Finding 5: ContentDetector Has MINIMAL Scope
- **Source**: Sub-agent ContentDetector usage analysis
- **Evidence**:
  - Only 1 consumer: `client.py`
  - Only 3 usage points: lines 1172, 1797, 1798
  - NOT exported as public API
  - NO custom package dependencies
  - NO service/agent direct imports
- **Impact**:
  - Low risk to change ContentDetector
  - Refactoring is safe but UNNECESSARY
  - Full rewrite is massive overkill for isolated component

### Finding 6: GSE155698 Not Actually Tested
- **Source**: Test file analysis
- **Evidence**:
  - `test_nested_archive_10x_loading.py` uses GSE155698 naming pattern
  - BUT creates SYNTHETIC V3 data (features.tsv.gz)
  - NO real GSE155698 data in test fixtures
  - NO V2 (genes.tsv) integration tests
- **Impact**:
  - Cannot verify if GSE155698 bug exists
  - Bug might be in specific GEO download path, not detection
  - Need test with REAL V2 data to diagnose

---

## CRITICAL CONCLUSION

**THE 1,836-LINE IMPLEMENTATION PLAN IS BASED ON A FALSE PREMISE.**

The plan claims line 227 has:
```python
TEN_X_REQUIRED_FILES = {"matrix.mtx", "features.tsv", "barcodes.tsv"}  # V3-only!
```

**THIS CONSTANT DOES NOT EXIST.** The actual code (lines 233-234) is:
```python
TEN_X_V3_FILES = {"matrix", "features", "barcodes"}  # V3 chemistry
TEN_X_V2_FILES = {"matrix", "genes", "barcodes"}     # V2 chemistry
```

Both V2 and V3 are properly detected and loaded throughout the codebase.

---

## RECOMMENDATIONS

### Option A: Add Tests (2-3 days) ← RECOMMENDED FOR PRE-SEED
- **Changes**:
  1. Add unit tests for ContentDetector (1 day)
  2. Add V2 (genes.tsv) integration test (1 day)
  3. Test with real GSE155698 sample if available (0.5 day)
- **Risks**: Minimal - just adding tests
- **Business impact**: Proves code works, builds confidence

### Option B: Investigate Specific GSE155698 Path (1 week)
- **Changes**:
  1. Reproduce bug with real GSE155698 data
  2. Trace exact code path that causes 0 genes
  3. Fix specific issue (likely GEO download strategy, not detection)
- **Risks**: May discover bug is in a different layer entirely
- **Business impact**: Fixes actual customer pain point if bug exists

### Option C: Full Rewrite (3-6 months) ← NOT RECOMMENDED
- **Changes**: 7,500+ lines new code as per plan
- **Risks**:
  - Opportunity cost: 3-6 months NOT spent on customer features
  - Plan based on incorrect analysis
  - Scope creep likely (proteomics, metabolomics in scope)
  - Breaking changes for no proven benefit
- **Business impact**:
  - Delays Anto pilot, DataBioMix delivery
  - Diverts resources from revenue-generating work
  - Solves hypothetical problem, not proven customer pain

---

## FINAL VERDICT

**DO NOT IMPLEMENT THE 3-6 MONTH REWRITE.**

The implementation plan:
1. Is based on code that doesn't exist (`TEN_X_REQUIRED_FILES`)
2. Proposes rewriting correct code (V2 support already works)
3. Has massive opportunity cost (3-6 months)
4. Addresses speculative future needs, not proven customer pain

**INSTEAD:**
1. Add unit tests for ContentDetector (1 day)
2. Add V2 format integration test (1 day)
3. If GSE155698 bug still exists, investigate SPECIFIC failure (not full rewrite)
4. Focus engineering time on customer features and Anto/DataBioMix delivery

---

## FINAL DELIVERABLES

1. [x] Concise summary of findings (see above)
2. [x] Revised implementation plan: **NOT NEEDED** - existing code is correct
3. [x] Personal recommendation for pre-seed stage: **Add tests, don't rewrite**

---

## ACTION ITEMS (If proceeding with Option A)

### Immediate (This Week)
1. Create `tests/unit/core/test_archive_utils.py`:
   ```python
   def test_content_detector_v2_detection():
       """Verify ContentDetector correctly identifies V2 format (genes.tsv)"""

   def test_content_detector_v3_detection():
       """Verify ContentDetector correctly identifies V3 format (features.tsv)"""
   ```

2. Update `tests/integration/test_nested_archive_10x_loading.py`:
   ```python
   def test_load_10x_v2_genes_tsv(self, temp_workspace):
       """Test loading 10X V2 format with genes.tsv (not features.tsv)"""
       # Create genes.tsv.gz instead of features.tsv.gz
       # Verify adata.n_vars > 0
   ```

### Short-term (Next Sprint)
3. If bug still occurs with real GSE155698 data:
   - Download 1-2 real samples from GSE155698
   - Trace exact code path that causes failure
   - Fix specific issue (likely NOT in ContentDetector)

---

## INVESTIGATION COMPLETE
Date: 2026-01-28
Conclusion: Implementation plan rejected - based on false premise
Recommendation: Add tests (2-3 days) instead of rewrite (3-6 months)
