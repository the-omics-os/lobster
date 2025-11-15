# Phase 1 SRA Provider - Bug Fix Reference

This document provides code fixes for the critical bugs found during Phase 1 QA testing.

---

## Bug #1: Empty Results Crash (CRITICAL)

**Location**: `/Users/tyo/GITHUB/omics-os/lobster/lobster/tools/providers/sra_provider.py:211-214`

**Severity**: BLOCKER

**Issue**: When NCBI returns no results or malformed XML, `xmltodict.parse()` returns `None`, causing AttributeError when calling `.get()` on None.

### Current Code (BROKEN)

```python
def _ncbi_esearch(
    self, query: str, filters: Dict[str, Any], max_results: int
) -> List[str]:
    """Execute NCBI esearch to get SRA IDs."""
    try:
        # ... URL building code ...

        # Make request
        response = urllib.request.urlopen(url, context=self.ssl_context, timeout=30)
        content = response.read()

        # Parse XML
        result = self.parse_xml(content)

        # Extract IDs - CRASHES HERE IF result IS None
        id_list = result.get("eSearchResult", {}).get("IdList", {}).get("Id", [])

        # Handle single ID case
        if isinstance(id_list, str):
            id_list = [id_list]
        elif id_list is None:
            id_list = []

        logger.debug(f"Found {len(id_list)} SRA IDs from esearch")
        return id_list
```

### Fixed Code

```python
def _ncbi_esearch(
    self, query: str, filters: Dict[str, Any], max_results: int
) -> List[str]:
    """Execute NCBI esearch to get SRA IDs."""
    try:
        # ... URL building code ...

        # Make request
        response = urllib.request.urlopen(url, context=self.ssl_context, timeout=30)
        content = response.read()

        # Parse XML
        result = self.parse_xml(content)

        # CRITICAL FIX: Handle None result from XML parsing
        if result is None:
            logger.warning(f"NCBI esearch returned None/invalid XML for query: {query}")
            return []

        # Extract IDs
        id_list = result.get("eSearchResult", {}).get("IdList", {}).get("Id", [])

        # Handle single ID case
        if isinstance(id_list, str):
            id_list = [id_list]
        elif id_list is None:
            id_list = []

        logger.debug(f"Found {len(id_list)} SRA IDs from esearch")
        return id_list
```

### Test Case

Add this test to verify the fix:

```python
def test_empty_results_no_crash(sra_provider):
    """Test that nonsensical queries don't crash."""
    result = sra_provider.search_publications("zzz_nonexistent_12345", max_results=3)

    # Should return graceful message, not crash
    assert "No" in result
    assert "Results Found" in result or "datasets found" in result
    assert len(result) > 50  # Should have helpful message
```

---

## Bug #2: Missing Organism/Platform in Accession Lookups (MAJOR)

**Location**: `/Users/tyo/GITHUB/omics-os/lobster/lobster/tools/providers/sra_provider.py:636-654`

**Severity**: MAJOR

**Issue**: When looking up accessions via pysradb.sra_metadata() (Path 1), the results often lack organism and platform fields, even with `detailed=True`. Path 2 (direct NCBI API) has these fields.

### Current Code (Inconsistent)

```python
def search_publications(
    self,
    query: str,
    max_results: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    try:
        db = self._get_sraweb()

        # Check if query is SRA accession pattern
        if self._is_sra_accession(query):
            # Path 1: Direct metadata retrieval for accession (INCOMPLETE METADATA)
            detailed = kwargs.get("detailed", True)
            df = db.sra_metadata(query, detailed=detailed)

            # ... rest of code uses pysradb results
```

### Fix Option A: Use Direct NCBI API for Accessions (RECOMMENDED)

This ensures BOTH paths use the same data source with complete metadata.

```python
def search_publications(
    self,
    query: str,
    max_results: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    """
    Search SRA database using direct NCBI API for both accessions and keywords.

    This method now uses direct NCBI API (esearch + esummary) for ALL queries,
    ensuring consistent and complete metadata.
    """
    try:
        # Check if query is SRA accession pattern
        if self._is_sra_accession(query):
            # Path 1 (FIXED): Use direct NCBI API for accessions too
            logger.info(f"[Phase 1] Performing direct NCBI SRA accession lookup: {query}")

            try:
                # Accessions go directly to esummary (no esearch needed)
                # Build mock esearch result with just this accession's ID
                # OR: Use esearch with the accession as query

                # Option 1: Query NCBI for the accession ID first
                ncbi_query = query  # Accession itself is the query
                sra_ids = self._ncbi_esearch(ncbi_query, {}, max_results)

                if not sra_ids:
                    return (
                        f"## No SRA Results Found\n\n"
                        f"**Query**: `{query}`\n\n"
                        f"No metadata found for this SRA accession. "
                        f"Verify the accession is valid and publicly available."
                    )

                # Fetch metadata via esummary (consistent with Path 2)
                df = self._ncbi_esummary(sra_ids)

                if df.empty:
                    return (
                        f"## No SRA Results Found\n\n"
                        f"**Query**: `{query}`\n\n"
                        f"No metadata found for this SRA accession."
                    )

                # Apply filters if provided
                if filters:
                    df = self._apply_filters(df, filters)

                return self._format_search_results(df, query, max_results, filters)

            except SRAProviderError:
                raise
            except Exception as e:
                logger.error(f"Direct NCBI SRA accession lookup error: {e}")
                raise SRAProviderError(
                    f"Error looking up SRA accession via NCBI API: {str(e)}"
                ) from e

        else:
            # Path 2: Direct NCBI esearch (unchanged)
            logger.info(f"[Phase 1] Performing direct NCBI SRA search: {query[:50]}...")

            # ... rest of existing Path 2 code ...
```

### Fix Option B: Supplement pysradb Results (Alternative)

Keep pysradb for accessions but supplement with NCBI API for missing fields.

```python
def search_publications(
    self,
    query: str,
    max_results: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    try:
        db = self._get_sraweb()

        if self._is_sra_accession(query):
            # Path 1: pysradb for basic data
            detailed = kwargs.get("detailed", True)
            df = db.sra_metadata(query, detailed=detailed)

            if df is None or df.empty:
                return (...)

            # SUPPLEMENT: If organism or platform missing, fetch from NCBI
            if df is not None and not df.empty:
                missing_organism = "organism" not in df.columns or df["organism"].isna().all()
                missing_platform = "instrument_platform" not in df.columns or df["instrument_platform"].isna().all()

                if missing_organism or missing_platform:
                    logger.info("Supplementing pysradb results with NCBI API for missing fields")

                    # Get SRA IDs for NCBI lookup
                    sra_ids = self._ncbi_esearch(query, {}, max_results)

                    if sra_ids:
                        ncbi_df = self._ncbi_esummary(sra_ids)

                        # Merge missing fields
                        if not ncbi_df.empty:
                            # Match by accession and copy missing fields
                            # (Implementation depends on DataFrame structure)
                            df = self._merge_metadata(df, ncbi_df)

            # Apply filters
            if filters:
                df = self._apply_filters(df, filters)

            return self._format_search_results(df, query, max_results, filters)
```

### Recommendation

**Use Option A** (direct NCBI API for all queries) because:
1. Consistent data source for both paths
2. Simpler code (no merging logic)
3. More complete metadata guaranteed
4. Already implemented and tested for Path 2

### Test Case

Add this test to verify organism/platform presence:

```python
def test_accession_has_complete_metadata(sra_provider):
    """Test that accession lookups have organism and platform fields."""
    result = sra_provider.search_publications("SRP033351", max_results=3)

    # Check for required metadata fields
    assert "Organism:" in result or "organism" in result.lower(), "Missing organism field"
    assert "Platform:" in result or "platform" in result.lower(), "Missing platform field"
    assert "Strategy:" in result or "strategy" in result.lower(), "Missing strategy field"
    assert "Layout:" in result or "layout" in result.lower(), "Missing layout field"
```

---

## Testing After Fixes

### 1. Run Manual Validation

```bash
cd /Users/tyo/GITHUB/omics-os/lobster
source .venv/bin/activate
python test_phase1_validation.py
```

**Expected**: Pass rate ≥90% (9/10 tests)

### 2. Run Integration Tests

```bash
pytest tests/integration/test_sra_provider_phase1.py \
  -k "not (rate_limiter or sequential)" \
  -v --tb=short
```

**Expected**: All functional tests pass (rate limit failures acceptable)

### 3. Test Specific Bugs

**Bug #1 (Empty Results)**:
```bash
python -c "
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.sra_provider import SRAProvider

dm = DataManagerV2()
provider = SRAProvider(dm)
result = provider.search_publications('zzz_nonexistent_12345', max_results=3)
assert 'No' in result and 'Results Found' in result
print('✅ Bug #1 FIXED')
"
```

**Bug #2 (Missing Metadata)**:
```bash
python -c "
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.sra_provider import SRAProvider

dm = DataManagerV2()
provider = SRAProvider(dm)
result = provider.search_publications('SRP033351', max_results=3)
assert 'Organism:' in result or 'organism' in result.lower()
assert 'Platform:' in result or 'platform' in result.lower()
print('✅ Bug #2 FIXED')
"
```

---

## Commit Message Template

```
fix(sra_provider): resolve Phase 1 critical bugs

Bug #1: Empty results crash
- Add null check after XML parsing in _ncbi_esearch()
- Return empty list when NCBI returns invalid/no results
- Prevents AttributeError on edge case queries

Bug #2: Missing organism/platform in accession lookups
- Switch accession lookups to direct NCBI API (consistent with keyword search)
- Ensures complete metadata for both Path 1 and Path 2
- Improves user experience with richer dataset information

Testing:
- Manual validation: 9/10 tests pass (90%)
- Integration tests: All functional tests pass
- Edge cases: Empty results, invalid accessions handled gracefully

Closes #XXX (QA Phase 1 Blockers)
```

---

## Rollout Plan

1. **Developer**: Apply fixes (2-4 hours)
2. **QA**: Re-run validation suite (30 minutes)
3. **Code Review**: Review fixes (30 minutes)
4. **Merge**: Merge to dev branch
5. **Staging**: Deploy to staging, smoke test (1 hour)
6. **Production**: Deploy Phase 1 to production
7. **Monitor**: Watch error rates, performance for 24 hours

---

## Success Criteria (Re-test)

- [ ] Manual validation pass rate ≥90% (9/10 tests)
- [ ] Integration tests: All functional tests pass
- [ ] Bug #1 fixed: Empty results return graceful message
- [ ] Bug #2 fixed: Accession lookups have organism + platform fields
- [ ] Performance: Accession search <3s (acceptable)
- [ ] No new bugs introduced

---

## Contact

Questions about these fixes? See:
- Full QA Report: `PHASE1_QA_REPORT.md`
- Test Outputs: `PHASE1_TEST_OUTPUTS.md`
- Executive Summary: `PHASE1_EXECUTIVE_SUMMARY.md`
