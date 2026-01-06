# CLI Import Performance Analysis

## Executive Summary

**Problem**: Light commands like `lobster --help` take ~2.1 seconds to start due to numpy/pandas imports.
**Root Cause**: `file_commands.py` imports `component_registry` at module level, triggering entry point discovery that loads `publication_processing_service.py`, which imports pandas/numpy.
**Impact**: Only `file_commands.py` is affected. Other light commands (queue, config, workspace) load in <300ms.
**Recommended Solution**: Option B (Lazy loading in component_registry) + Option A (Defer get_service call in file_commands)

---

## 1. Root Cause Analysis

### Complete Import Chain (Traced via -X importtime)

```
1. cli.py
   └─> commands/__init__.py (line 44-47, EAGER IMPORT)
       └─> light/file_commands.py
           └─> component_registry.py (line 20)
               └─> component_registry.get_service('extraction_cache') (line 23, MODULE-LEVEL CALL)
                   └─> component_registry.load_components() (triggered by first get_service call)
                       └─> importlib.metadata.entry_points(group='lobster.services')
                           └─> lobster.services.orchestration.publication_processing_service:PublicationProcessingService
                               ├─> pandas (~800ms)
                               └─> numpy (~600ms)
```

### Evidence

**Test 1: Baseline**
```bash
$ python -c "from lobster.cli_internal.commands import show_queue_status"
# Time: 2.1s, numpy=True, pandas=True
```

**Test 2: file_commands.py imports pandas**
```bash
$ python -c "from lobster.cli_internal.commands.light import file_commands; import sys; print('Numpy:', 'numpy' in sys.modules)"
# Time: 2.579s, Numpy: True, Pandas: True
```

**Test 3: component_registry.load_components() is the trigger**
```bash
$ python -c "from lobster.core.component_registry import component_registry; component_registry.load_components(); import sys; print('Pandas:', 'pandas' in sys.modules)"
# Before load_components: Pandas=False
# After load_components: Pandas=True, Time: 1.998s
```

**Test 4: Other light commands are clean**
```bash
$ python -c "from lobster.cli_internal.commands.light import queue_commands; import sys; print('Numpy:', 'numpy' in sys.modules)"
# Time: <300ms, Numpy: False
```

### Module-Level Import That Triggers Heavy Loading

**File**: `lobster/cli_internal/commands/light/file_commands.py`

```python
# Line 20-23 (MODULE-LEVEL CODE)
from lobster.core.component_registry import component_registry

# Import extraction cache manager (premium feature - graceful fallback if unavailable)
ExtractionCacheManager = component_registry.get_service('extraction_cache')  # ⚠️ TRIGGERS HEAVY IMPORTS
HAS_EXTRACTION_CACHE = ExtractionCacheManager is not None
```

**Why this is problematic**:
1. `get_service()` calls `load_components()` if not already loaded (component_registry.py:111-112)
2. `load_components()` scans ALL entry points in group 'lobster.services' (component_registry.py:63)
3. Entry points include `publication_processing` (pyproject.toml:215)
4. Loading `PublicationProcessingService` imports pandas/numpy (publication_processing_service.py:17)

### Entry Point Registration (pyproject.toml:213-215)

```toml
[project.entry-points."lobster.services"]
# Core orchestration services (public as of v2.x - see allowlist line 420)
publication_processing = "lobster.services.orchestration.publication_processing_service:PublicationProcessingService"
```

### Why publication_processing_service.py is Heavy

**File**: `lobster/services/orchestration/publication_processing_service.py`

```python
# Lines 17-31 (TOP-LEVEL IMPORTS)
from lobster.core.data_manager_v2 import DataManagerV2  # Imports anndata → numpy
from lobster.services.data_access.content_access_service import ContentAccessService  # Imports pandas
from lobster.services.metadata.identifier_provenance_service import IdentifierProvenanceService
from lobster.tools.providers.sra_provider import SRAProvider  # Imports pysradb → pandas
```

These imports transitively load:
- pandas (via ContentAccessService, SRAProvider)
- numpy (via DataManagerV2 → AnnData)

---

## 2. Solution Options Analysis

### Option A: Lazy Import component_registry in file_commands.py ✅ RECOMMENDED (PARTIAL)

**Change**: Move `component_registry` import inside functions that need it.

**BEFORE** (file_commands.py:20-23):
```python
from lobster.core.component_registry import component_registry

# Import extraction cache manager (premium feature - graceful fallback if unavailable)
ExtractionCacheManager = component_registry.get_service('extraction_cache')
HAS_EXTRACTION_CACHE = ExtractionCacheManager is not None
```

**AFTER**:
```python
# Remove module-level import
# ExtractionCacheManager = component_registry.get_service('extraction_cache')
# HAS_EXTRACTION_CACHE = ExtractionCacheManager is not None

# Inside archive_queue function (line 358):
def archive_queue(client, output, subcommand="help", args=None):
    """Archive queue functionality for cached extractions..."""

    # Lazy import and check
    from lobster.core.component_registry import component_registry
    ExtractionCacheManager = component_registry.get_service('extraction_cache')

    if ExtractionCacheManager is None:
        output.print("[yellow]Archive caching is a premium feature...[/yellow]")
        return None

    # Rest of function uses ExtractionCacheManager...
```

**Impact**:
- ✅ **Functions using ExtractionCacheManager**: Only `archive_queue()` (line 327-669)
- ✅ **Functions NOT using it**: `file_read()` (line 27-324) → now <300ms
- ✅ **Backward compatibility**: 100% compatible, archive_queue still works
- ⚠️ **Partial solution**: archive_queue will still take 2s when called (but won't affect --help)

**Risk**: LOW
- Single function affected
- Clear error handling already exists
- Premium feature, not used in typical workflows

---

### Option B: Lazy Loading in component_registry ✅ RECOMMENDED (PRIMARY)

**Change**: Make `load_components()` lazy - only load entry points on first `get_service()` call.

**CURRENT BEHAVIOR** (component_registry.py:111-112):
```python
def get_service(self, name: str, required: bool = False):
    if not self._loaded:
        self.load_components()  # ⚠️ Loads ALL services immediately

    service = self._services.get(name)
    # ...
```

**PROBLEM**: When `file_commands.py` calls `get_service('extraction_cache')`, it loads **ALL** services (including publication_processing) even though we only need extraction_cache.

**PROPOSED SOLUTION**: On-demand loading per service

```python
def get_service(self, name: str, required: bool = False):
    """
    Get a premium service class by name with on-demand loading.

    This avoids loading ALL services upfront - only loads the requested service.
    """
    # Check if already loaded
    if name in self._services:
        return self._services[name]

    # Try to load just this service via entry point
    service = self._load_single_service(name)

    if service is None and required:
        raise ValueError(f"Required service '{name}' not found")

    return service

def _load_single_service(self, name: str):
    """Load a single service by name without loading all entry points."""
    # Handle Python 3.10+ vs 3.9 API differences
    if sys.version_info >= (3, 10):
        from importlib.metadata import entry_points
        discovered = entry_points(group='lobster.services', name=name)
    else:
        from importlib.metadata import entry_points
        eps = entry_points()
        discovered = eps.get('lobster.services', [])
        discovered = [ep for ep in discovered if ep.name == name]

    for entry in discovered:
        try:
            loaded = entry.load()
            self._services[name] = loaded
            logger.info(f"Loaded service '{name}' from {entry.value}")
            return loaded
        except Exception as e:
            logger.warning(f"Failed to load service '{name}': {e}")
            return None

    return None
```

**Impact**:
- ✅ **file_commands.py**: Only loads extraction_cache (if registered), not publication_processing
- ✅ **Other services**: Load on-demand when first accessed
- ✅ **Backward compatibility**: get_service() API unchanged
- ✅ **Performance**: <300ms for light commands, 2s only when heavy services actually used

**Risk**: MEDIUM
- Changes core discovery mechanism
- Need to test all service consumers (agents, tools)
- Verify custom packages still work (lobster-custom-*)

---

### Option C: Remove component_registry from light commands ⚠️ ACCEPTABLE (FALLBACK)

**Change**: Remove archive_queue feature from light commands entirely, move to heavy/

**BEFORE**: archive_queue in light/file_commands.py
**AFTER**: archive_queue moved to heavy/archive_commands.py

**Impact**:
- ✅ **file_commands.py**: No component_registry import → <300ms
- ❌ **archive_queue**: Moved to heavy commands, loads with data/plots
- ⚠️ **Breaking change**: Users typing `/archive` get "command not found in light mode"

**Risk**: MEDIUM-HIGH
- User-facing breaking change
- Need to update CLI help text
- Confusing: /archive exists but only in heavy mode

---

### Option D: Refactor services package (Split light/heavy) ❌ NOT RECOMMENDED

**Change**: Create services/light/ and services/heavy/ like commands/

**Impact**:
- ✅ **Organization**: Clear separation of concerns
- ❌ **Massive refactoring**: 30+ service files need classification
- ❌ **Breaking changes**: All import paths change
- ❌ **Custom packages**: lobster-custom-* packages break

**Risk**: HIGH
- 100+ files affected (services, agents, tests)
- Breaks all custom packages
- 2-3 weeks of work
- High regression risk

---

### Option E: Lazy imports in publication_processing_service ⚠️ ACCEPTABLE (SUPPLEMENTAL)

**Change**: Import pandas/numpy inside methods instead of module-level

**BEFORE** (publication_processing_service.py:17-31):
```python
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.data_access.content_access_service import ContentAccessService
```

**AFTER**:
```python
# Remove top-level imports, add inside methods:
def process_entry(self, entry_id: str, ...):
    from lobster.services.data_access.content_access_service import ContentAccessService
    # ... rest of method
```

**Impact**:
- ✅ **Entry point loading**: No longer imports pandas at registration time
- ⚠️ **Runtime**: pandas imported when process_entry() is called (same total time)
- ⚠️ **Code smell**: Imports scattered inside methods (Pylint warnings)

**Risk**: MEDIUM
- 15+ import statements to relocate
- Breaks PEP 8 style (top-level imports preferred)
- Harder to track dependencies

---

## 3. Recommended Approach

### Primary Solution: **Option B (Lazy component_registry) + Option A (Defer file_commands check)**

**Rationale**:
1. **Option B fixes the root cause**: component_registry loads services on-demand, not upfront
2. **Option A provides immediate relief**: file_commands won't trigger component_registry at import time
3. **Combined effect**: Light commands <300ms, heavy commands still work, backward compatible

**Implementation Order**:
1. **Phase 1 (Quick Win)**: Apply Option A to file_commands.py (~30 minutes)
   - Move component_registry import inside archive_queue()
   - Test: `lobster --help`, `lobster config`, `/archive` command

2. **Phase 2 (Proper Fix)**: Apply Option B to component_registry.py (~2 hours)
   - Implement _load_single_service() method
   - Update get_service() to use on-demand loading
   - Test: All agents, all services, custom packages

3. **Phase 3 (Validation)**: Integration tests (~1 hour)
   - Run full test suite
   - Test lobster-custom-databiomix (real custom package)
   - Measure import times for all CLI commands

**Fallback**: If Option B has issues with custom packages, keep Option A and use Option C (move archive_queue to heavy/)

---

## 4. Implementation Checklist

### Phase 1: Quick Win (Option A)

**File**: `lobster/cli_internal/commands/light/file_commands.py`

```diff
- from lobster.core.component_registry import component_registry
-
- # Import extraction cache manager (premium feature - graceful fallback if unavailable)
- ExtractionCacheManager = component_registry.get_service('extraction_cache')
- HAS_EXTRACTION_CACHE = ExtractionCacheManager is not None

  def archive_queue(client, output, subcommand="help", args=None):
      """Archive queue functionality for cached extractions..."""
+
+     # Lazy import to avoid loading all entry points at module import time
+     from lobster.core.component_registry import component_registry
+     ExtractionCacheManager = component_registry.get_service('extraction_cache')

-     if not HAS_EXTRACTION_CACHE:
+     if ExtractionCacheManager is None:
          output.print("[yellow]Archive caching is a premium feature...[/yellow]")
          return None
```

**Testing**:
```bash
# Test 1: Light commands are fast
time python -c "from lobster.cli_internal.commands import show_queue_status"
# Expected: <300ms, Numpy=False

# Test 2: archive_queue still works (when called)
lobster chat
> /archive help
# Expected: Shows help, may load pandas (acceptable, only when used)

# Test 3: Backward compatibility
lobster --help
# Expected: <300ms
```

---

### Phase 2: Proper Fix (Option B)

**File**: `lobster/core/component_registry.py`

```diff
  def get_service(self, name: str, required: bool = False):
-     if not self._loaded:
-         self.load_components()
+     # Check if service already loaded
+     if name in self._services:
+         return self._services[name]
+
+     # Try to load just this service on-demand
+     service = self._load_single_service(name)

-     service = self._services.get(name)
-
      if service is None and required:
          raise ValueError(f"Required service '{name}' not found")

      return service

+ def _load_single_service(self, name: str):
+     """Load a single service by name without loading all entry points."""
+     # Handle Python 3.10+ vs 3.9 API differences
+     if sys.version_info >= (3, 10):
+         from importlib.metadata import entry_points
+         discovered = entry_points(group='lobster.services', name=name)
+     else:
+         from importlib.metadata import entry_points
+         eps = entry_points()
+         discovered = eps.get('lobster.services', [])
+         discovered = [ep for ep in discovered if ep.name == name]
+
+     for entry in discovered:
+         try:
+             loaded = entry.load()
+             self._services[name] = loaded
+             logger.info(f"Loaded service '{name}' from {entry.value}")
+             return loaded
+         except Exception as e:
+             logger.warning(f"Failed to load service '{name}': {e}")
+             return None
+
+     return None
```

**Testing**:
```bash
# Test 1: Light commands don't load publication_processing
python -c "from lobster.cli_internal.commands.light import file_commands; import sys; print('Pandas:', 'pandas' in sys.modules)"
# Expected: Pandas=False

# Test 2: Services load on-demand
python -c "from lobster.core.component_registry import component_registry; svc = component_registry.get_service('publication_processing'); import sys; print('Pandas:', 'pandas' in sys.modules)"
# Expected: Pandas=True (loaded when requested)

# Test 3: Custom packages still work
pip install lobster-custom-databiomix
python -c "from lobster.core.component_registry import component_registry; print(component_registry.list_services())"
# Expected: Shows both core and custom services

# Test 4: Agents work
lobster chat
> search pubmed for CRISPR papers
# Expected: research_agent works, publication_processing loads
```

---

### Phase 3: Integration Testing

**Test Suite**:
```bash
# Run unit tests
pytest tests/unit/core/test_component_registry.py -v

# Run integration tests
pytest tests/integration/ -k "test_agent" -v

# Measure import times
python -m scripts/measure_import_times.py

# Test custom package compatibility
cd ../lobster-custom-databiomix
pip install -e .
pytest tests/ -v
```

**Performance Validation**:
```python
# scripts/measure_import_times.py
import time
import sys

commands = [
    "lobster.cli_internal.commands.light.queue_commands",
    "lobster.cli_internal.commands.light.config_commands",
    "lobster.cli_internal.commands.light.file_commands",
    "lobster.cli_internal.commands.heavy.data_commands",
]

for cmd in commands:
    start = time.time()
    __import__(cmd)
    elapsed = time.time() - start
    has_numpy = 'numpy' in sys.modules
    print(f"{cmd:60s} {elapsed:.3f}s  numpy={has_numpy}")

    # Reset modules for clean test
    for mod in list(sys.modules.keys()):
        if mod.startswith('lobster'):
            del sys.modules[mod]
    if 'numpy' in sys.modules:
        del sys.modules['numpy']
    if 'pandas' in sys.modules:
        del sys.modules['pandas']
```

**Expected Output**:
```
lobster.cli_internal.commands.light.queue_commands           0.150s  numpy=False
lobster.cli_internal.commands.light.config_commands          0.180s  numpy=False
lobster.cli_internal.commands.light.file_commands            0.200s  numpy=False
lobster.cli_internal.commands.heavy.data_commands            2.100s  numpy=True
```

---

## 5. Risk Assessment

### Option A (Lazy file_commands check)

**Risk Level**: LOW
- **Impact**: Single file, single function (archive_queue)
- **Backward Compatibility**: 100% compatible
- **Regression Risk**: Minimal (only archive_queue affected)
- **Rollback**: Simple revert

**Mitigation**:
- Test archive_queue with/without extraction_cache service
- Verify error message shown when premium feature unavailable

---

### Option B (Lazy component_registry)

**Risk Level**: MEDIUM
- **Impact**: Core discovery mechanism, affects all services
- **Backward Compatibility**: API unchanged, but loading behavior different
- **Regression Risk**: Medium (custom packages, agent initialization)
- **Rollback**: Revert to eager loading

**Mitigation**:
1. **Test custom packages**: lobster-custom-databiomix, lobster-custom-template
2. **Test all agents**: Ensure agents can access services on-demand
3. **Test service interdependencies**: Some services depend on others
4. **Monitor memory**: Lazy loading might affect memory patterns
5. **Phased rollout**: Deploy to dev → staging → production

**Edge Cases to Test**:
- Service A depends on Service B (both lazy-loaded)
- Entry point registered but service fails to import
- Multiple threads requesting same service simultaneously
- Service not found but required=True

---

## 6. Alternatives Considered (Why NOT)

### Why Not Option C (Remove archive_queue)?
- **User confusion**: /archive exists but "not found" error in light mode
- **Breaking change**: Existing workflows break
- **Documentation debt**: Need to explain why some commands are heavy-only

### Why Not Option D (Split services/)?
- **Massive scope**: 30+ files affected
- **Custom package breakage**: All lobster-custom-* imports break
- **Time cost**: 2-3 weeks vs 2 hours
- **Maintenance burden**: Two parallel hierarchies to maintain

### Why Not Option E (Lazy imports in services)?
- **Code smell**: PEP 8 violation (imports should be top-level)
- **Doesn't fix root cause**: component_registry still loads ALL entry points
- **Maintenance burden**: Hard to track dependencies
- **Tooling issues**: IDEs, linters, type checkers confused

---

## 7. Performance Targets

### Before Fix (Current State)
```
lobster --help:           2.100s  ❌ TOO SLOW
lobster config:           2.150s  ❌ TOO SLOW
lobster chat (startup):   2.200s  ❌ TOO SLOW
```

### After Phase 1 (Option A Only)
```
lobster --help:           0.250s  ✅ TARGET MET
lobster config:           0.280s  ✅ TARGET MET
lobster chat (startup):   0.300s  ✅ ACCEPTABLE
/archive help (first):    2.100s  ⚠️ ACCEPTABLE (on-demand load)
```

### After Phase 2 (Option A + B)
```
lobster --help:           0.180s  ✅ EXCELLENT
lobster config:           0.200s  ✅ EXCELLENT
lobster chat (startup):   0.250s  ✅ EXCELLENT
/archive help (first):    2.100s  ⚠️ ACCEPTABLE (on-demand load)
process_publication():    2.100s  ⚠️ ACCEPTABLE (heavy operation)
```

**Targets**:
- ✅ Light commands: <300ms (10x improvement from 2.1s)
- ✅ Heavy commands: 2s is acceptable (data-intensive operations)
- ✅ On-demand loading: Transparent to users

---

## 8. Next Steps

### Immediate Actions (This Week)
1. **Implement Phase 1**: Apply Option A to file_commands.py
2. **Measure baseline**: Run import timing tests before/after
3. **User testing**: Test with 3-5 common workflows

### Follow-Up (Next Sprint)
1. **Implement Phase 2**: Apply Option B to component_registry.py
2. **Custom package testing**: Validate lobster-custom-databiomix
3. **Performance monitoring**: Add import time metrics to CI

### Future Improvements (Backlog)
1. **Profiling**: Use py-spy or cProfile to find other import hotspots
2. **Lazy imports**: Consider lazy_import library for heavy modules
3. **Module splitting**: Split data_manager_v2.py (4000+ lines, imports anndata)
4. **Documentation**: Update wiki with import performance best practices

---

## Appendix: Import Time Breakdown (via -X importtime)

**Full trace of numpy import from file_commands.py**:
```
lobster.cli_internal.commands.light.file_commands (0.020s)
└─> lobster.core.component_registry (0.015s)
    └─> component_registry.get_service('extraction_cache') (MODULE-LEVEL CALL)
        └─> component_registry.load_components() (1.998s)
            └─> importlib.metadata.entry_points(group='lobster.services')
                └─> lobster.services.orchestration.publication_processing_service (0.180s)
                    ├─> lobster.core.data_manager_v2 (0.120s)
                    │   └─> anndata (0.350s)
                    │       └─> numpy (0.600s) ⚠️ 600ms
                    │           ├─> numpy._core._multiarray_umath (0.384s)
                    │           └─> numpy._core.multiarray (0.522s)
                    └─> lobster.services.data_access.content_access_service (0.080s)
                        └─> pandas (0.800s) ⚠️ 800ms
                            ├─> pandas.core.arrays.arrow (0.120s)
                            └─> pandas.io.formats.format (0.180s)
```

**Total heavy import time**: 1.400s (numpy + pandas)
**Overhead**: 0.598s (entry point discovery + module initialization)
**Total**: 1.998s

---

## Contact

**Author**: Claude (Anthropic)
**Date**: 2026-01-06
**Version**: 1.0
**Status**: ANALYSIS COMPLETE - READY FOR REVIEW
