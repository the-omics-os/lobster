# Phase 5: Plugin-First Registration - Research

**Researched:** 2026-03-04
**Domain:** Python entry-point discovery, plugin registration pattern, pyproject.toml entry points
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PLUG-01 | Queue preparers discovered from `lobster.queue_preparers` entry points first-class | Entry-point group already wired in ComponentRegistry; pyproject.toml must add declarations |
| PLUG-02 | Download services discovered from `lobster.download_services` entry points first-class | Entry-point group already wired in ComponentRegistry; pyproject.toml must add declarations |
| PLUG-03 | Entry-point declarations added to pyproject.toml BEFORE fallback gating | Sequence: declare → verify → gate. Confirmed no declarations currently exist for these groups |
| PLUG-04 | All 5 databases (GEO, SRA, PRIDE, MassIVE, MetaboLights) discoverable via entry points | 5 queue preparer classes + 5 download service classes are ready; just need entry-point wiring in pyproject.toml |
| PLUG-05 | Existing tests updated for entry-point discovery assertions | Two test files use hardcoded-default assertions that must be rewritten to test EP path |
| PLUG-06 | Hardcoded fallback gated with explicit flag | Both routers already have a "Phase 2: Hardcoded fallback" block; add a single boolean gate |
</phase_requirements>

---

## Summary

Phase 5 is a focused mechanical refactoring with no algorithmic changes. The ComponentRegistry already has full discovery infrastructure for both `lobster.queue_preparers` and `lobster.download_services` entry-point groups. The two routers (`QueuePreparationService` and `DownloadOrchestrator`) already implement an entry-point discovery pass before their hardcoded fallback blocks. The only missing piece is that `pyproject.toml` in the core `lobster-ai` package has no declarations for these two groups, so the discovery pass silently yields zero results and falls through to the hardcoded block every time.

The work has four mechanical steps: (1) add entry-point declarations to `pyproject.toml`, (2) verify discovery works at runtime after `pip install -e .`, (3) gate the hardcoded fallback blocks behind an explicit boolean flag (defaulting to `False` once entry points are confirmed), and (4) update two test classes that assert hardcoded service/preparer types to assert on entry-point-discovered types instead.

**Primary recommendation:** Add `[project.entry-points."lobster.queue_preparers"]` and `[project.entry-points."lobster.download_services"]` sections to `pyproject.toml`, re-install the package, verify all 5 databases appear in `component_registry.list_queue_preparers()`, then gate the hardcoded fallback with `_ALLOW_HARDCODED_FALLBACK = False`.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `importlib.metadata` | stdlib | Entry-point discovery | PEP 420 / setuptools standard, already used by ComponentRegistry |
| `setuptools` | >=65.0 | Build backend — publishes entry points from pyproject.toml | Already the project build backend |
| `pytest` | >=7.0 | Test framework | Already in use; 43 tests pass in the affected files |
| `pytest-mock` / `unittest.mock` | stdlib | Mock entry point loading | Already used in test_component_registry.py |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `importlib_metadata` (backport) | n/a | Not needed — project requires Python 3.12+ which has stdlib version | Never needed here |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `importlib.metadata.entry_points(group=...)` | `pkg_resources.iter_entry_points()` | `pkg_resources` is legacy/slow; `importlib.metadata` is the current standard and is already used |

**Installation:** No new packages required. All tooling is already installed.

---

## Architecture Patterns

### Recommended Project Structure

No new files or folders are needed. Changes are confined to:

```
lobster/
├── pyproject.toml                                    # Add queue_preparers + download_services EP sections
└── lobster/
    ├── services/data_access/
    │   └── queue_preparation_service.py              # Add _ALLOW_HARDCODED_FALLBACK gate
    └── tools/
        └── download_orchestrator.py                  # Add _ALLOW_HARDCODED_FALLBACK gate
tests/
└── unit/
    └── services/data_access/
        ├── test_queue_preparation_service.py         # Update TestDefaultRegistration
        └── test_download_services.py                 # Update TestDownloadOrchestrator
```

### Pattern 1: pyproject.toml Entry-Point Declarations

**What:** Add two new `[project.entry-points.*]` sections declaring all 5 queue preparers and all 5 download services as entry points.

**When to use:** Required as PLUG-03/PLUG-04 — must be done FIRST before any fallback gating.

**Example:**
```toml
# Source: pyproject.toml – follows the existing lobster.services pattern at line 269

[project.entry-points."lobster.queue_preparers"]
geo         = "lobster.services.data_access.geo_queue_preparer:GEOQueuePreparer"
sra         = "lobster.services.data_access.sra_queue_preparer:SRAQueuePreparer"
pride       = "lobster.services.data_access.pride_queue_preparer:PRIDEQueuePreparer"
massive     = "lobster.services.data_access.massive_queue_preparer:MassIVEQueuePreparer"
metabolights = "lobster.services.data_access.metabolights_queue_preparer:MetaboLightsQueuePreparer"

[project.entry-points."lobster.download_services"]
geo         = "lobster.services.data_access.geo_download_service:GEODownloadService"
sra         = "lobster.services.data_access.sra_download_service:SRADownloadService"
pride       = "lobster.services.data_access.pride_download_service:PRIDEDownloadService"
massive     = "lobster.services.data_access.massive_download_service:MassIVEDownloadService"
metabolights = "lobster.services.data_access.metabolights_download_service:MetaboLightsDownloadService"
```

**Critical:** After modifying `pyproject.toml` the package must be re-installed (`uv pip install -e .`) for entry points to take effect. The existing `pip install -e .` or `make dev-install` path does this.

### Pattern 2: Hardcoded Fallback Gating

**What:** Both routers (`QueuePreparationService._register_default_preparers` and `DownloadOrchestrator._register_default_services`) have an identical two-phase structure: entry-point discovery first, then hardcoded fallback. The gate is a module-level constant.

**When to use:** Applied after PLUG-03/PLUG-04 are complete and verified.

**Example:**
```python
# Source: lobster/services/data_access/queue_preparation_service.py
# and lobster/tools/download_orchestrator.py — same pattern in both files

# Module-level constant — set to True ONLY for debugging/emergency recovery
_ALLOW_HARDCODED_FALLBACK = False


class QueuePreparationService:
    def _register_default_preparers(self) -> None:
        """
        Register default queue preparers for all supported databases.

        Entry-point discovery (lobster.queue_preparers) is the primary path.
        Hardcoded fallback is gated by _ALLOW_HARDCODED_FALLBACK and only
        activates when entry-point discovery yields nothing.
        """
        # Phase 1: Discover queue preparers from entry points
        discovered_names = set()
        try:
            from lobster.core.component_registry import component_registry
            for name, preparer_cls in component_registry.list_queue_preparers().items():
                try:
                    preparer = preparer_cls(self.data_manager)
                    self.register_preparer(preparer)
                    discovered_names.add(name)
                    logger.debug(f"Registered queue preparer '{name}' from entry point")
                except Exception as e:
                    logger.warning(
                        f"Failed to load queue preparer '{name}' from entry point: {e}"
                    )
        except Exception as e:
            logger.debug(f"Entry point queue preparer discovery skipped: {e}")

        # Phase 2: Hardcoded fallback — only when no entry-point discovery occurred
        if not _ALLOW_HARDCODED_FALLBACK and discovered_names:
            return  # Entry-point path succeeded — skip hardcoded fallback

        if not _ALLOW_HARDCODED_FALLBACK and not discovered_names:
            logger.warning(
                "No queue preparers discovered via entry points. "
                "Ensure pyproject.toml declares lobster.queue_preparers entry points "
                "and the package is installed (`uv pip install -e .`)."
            )
            return  # Still skip hardcoded fallback — force explicit opt-in

        # Hardcoded fallback (only active when _ALLOW_HARDCODED_FALLBACK=True)
        try:
            from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer
            self.register_preparer(GEOQueuePreparer(self.data_manager))
        except ImportError as e:
            logger.debug(f"GEOQueuePreparer not available: {e}")
        # ... repeat for other databases
```

**Note on gate semantics:** The gate condition in REQUIREMENTS says "hardcoded fallback only activates when entry-point discovery yields nothing." A simpler but equally valid interpretation: `_ALLOW_HARDCODED_FALLBACK = False` unconditionally skips the entire hardcoded block. This is the simpler/safer form and avoids any ambiguity about what "yields nothing" means during test isolation. Either semantics satisfies PLUG-06.

### Pattern 3: Entry-Point Discovery Tests

**What:** Tests that verify services/preparers are registered via the entry-point path, not hardcoded imports. Uses `importlib.metadata` mock or `dist_info` fixture.

**When to use:** PLUG-05 — replaces `TestDefaultRegistration` and `TestDownloadOrchestrator.test_auto_registration`.

**Example:**
```python
# Source: tests/unit/core/test_component_registry.py — existing mock_entry_points pattern

from unittest.mock import patch, Mock
import pytest
from lobster.services.data_access.queue_preparation_service import QueuePreparationService


def _make_mock_ep(name: str, cls):
    """Build a mock entry point that loads cls."""
    ep = Mock()
    ep.name = name
    ep.value = f"{cls.__module__}:{cls.__name__}"
    ep.dist = Mock()
    ep.dist.name = "lobster-ai"
    ep.load.return_value = cls
    return ep


class TestEntryPointDiscovery:
    """Validate that QueuePreparationService discovers preparers via entry points."""

    def test_geo_discovered_via_entry_point(self, mock_data_manager):
        from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer

        mock_ep = _make_mock_ep("geo", GEOQueuePreparer)

        with patch(
            "lobster.core.component_registry.ComponentRegistry._load_entry_point_group",
            side_effect=lambda group, target: (
                target.update({"geo": GEOQueuePreparer})
                if group == "lobster.queue_preparers"
                else None
            ),
        ):
            from lobster.core.component_registry import ComponentRegistry
            registry = ComponentRegistry()
            service = QueuePreparationService.__new__(QueuePreparationService)
            service.data_manager = mock_data_manager
            service._preparers = {}
            # Inject registry with known EP result
            with patch(
                "lobster.services.data_access.queue_preparation_service"
                ".component_registry.list_queue_preparers",
                return_value={"geo": GEOQueuePreparer}
            ):
                service._register_default_preparers()

            assert "geo" in service.list_supported_databases()
            preparer = service.get_preparer_for_database("geo")
            assert isinstance(preparer, GEOQueuePreparer)
```

**Simpler alternative (preferred):** Since all 5 services are in `lobster-ai` core with proper entry-point declarations, the tests can use the real installed package rather than mocking. After `uv pip install -e .` the entry points exist:

```python
def test_all_5_databases_discoverable_via_entry_points(self, mock_data_manager):
    """Verifies entry-point declarations in pyproject.toml are correct."""
    from importlib.metadata import entry_points

    qp_names = {ep.name for ep in entry_points(group="lobster.queue_preparers")}
    ds_names = {ep.name for ep in entry_points(group="lobster.download_services")}

    assert {"geo", "sra", "pride", "massive", "metabolights"} <= qp_names
    assert {"geo", "sra", "pride", "massive", "metabolights"} <= ds_names
```

This is a contract test — it validates `pyproject.toml` is correctly configured. Runs in < 1 second with no mocking needed.

### Anti-Patterns to Avoid

- **Removing hardcoded fallback before adding entry-point declarations:** If the hardcoded fallback is removed first and the `pyproject.toml` declarations are missing, ALL 5 databases vanish from routing. The sequence is: declare → reinstall → verify → gate. (PLUG-03 explicitly captures this.)
- **Gating fallback with `if not discovered_names`:** Looks reasonable but creates a silent failure mode — if discovery fails due to a stale install, the fallback silently activates. Better to log a warning and return, forcing explicit `_ALLOW_HARDCODED_FALLBACK = True` opt-in.
- **Using try/except ImportError around entry-point loading:** The ComponentRegistry already uses `except Exception` for each entry-point load failure. The routers should not add additional try/except around the registry call itself, or errors become invisible.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Entry-point discovery | Custom file-scanning or __import__ | `importlib.metadata.entry_points(group=...)` | PEP 420 standard; ComponentRegistry already uses it; handles editable installs, wheels, and namespace packages |
| Plugin contract validation | Custom reflection + attribute checks | `AgentContractTestMixin` pattern already in `testing/contract_mixins.py` | Existing contract test infrastructure handles compliance checks |

**Key insight:** The ComponentRegistry already solves all discovery complexity. Phase 5 is adding declarations to `pyproject.toml` and flipping a flag — not building a new system.

---

## Common Pitfalls

### Pitfall 1: Stale Entry Points After pyproject.toml Changes

**What goes wrong:** `component_registry.list_queue_preparers()` returns empty even after adding declarations.

**Why it happens:** Entry points are registered during package install/build. Editing `pyproject.toml` alone does nothing — the package dist-info must be regenerated.

**How to avoid:** After any `pyproject.toml` entry-point change, run `uv pip install -e .` (or `make dev-install`). Verify with:
```bash
python -c "
from importlib.metadata import entry_points
print(list(entry_points(group='lobster.queue_preparers')))
print(list(entry_points(group='lobster.download_services')))
"
```

**Warning signs:** `component_registry.list_queue_preparers()` returns `{}` after declaring entry points.

### Pitfall 2: Entry-Point Name Collision with Database Keys

**What goes wrong:** An entry-point name does not match the `supported_databases()` key of the class it loads, causing the router to register the service under the wrong key.

**Why it happens:** The entry-point name (e.g., `geo`) is NOT used as the registration key — the class's `supported_databases()` return value is. But if the entry-point name is `geo_preparer` and `supported_databases()` returns `["geo"]`, this is fine. Confusion arises when humans inspect the registry and see mismatched names.

**How to avoid:** Use the same string for the entry-point name as `supported_databases()[0]`. E.g., entry-point name `geo` for `GEOQueuePreparer.supported_databases() == ["geo"]`.

**Warning signs:** `service.list_supported_databases()` shows fewer than 5 databases after re-install.

### Pitfall 3: ComponentRegistry Singleton is Already Loaded When Tests Run

**What goes wrong:** `component_registry._loaded = True` and `_queue_preparers = {}` because the singleton was loaded before the test `patch()` is applied, so the mock never fires.

**Why it happens:** The `component_registry` singleton at module level loads on first access. If any previous test triggers `load_components()`, the state is cached.

**How to avoid:** Call `component_registry.reset()` in test teardown, OR use a fresh `ComponentRegistry()` instance (not the singleton) in tests. The existing `fresh_registry` fixture in `test_component_registry.py` already does this.

**Warning signs:** `component_registry.list_queue_preparers()` returns non-empty dict but with wrong keys.

### Pitfall 4: Test Asserts Hardcoded Type After Fallback is Gated

**What goes wrong:** `test_download_services.py::TestDownloadOrchestrator::test_service_lookup_geo` asserts `isinstance(service, GEODownloadService)` — this still passes because the service IS a `GEODownloadService` regardless of how it was discovered. This test does NOT break after gating.

**Why it happens:** The tests verify the correct class type, not HOW it was registered. So `test_auto_registration` (which asserts `"geo" in databases`) still passes. The issue is specifically `TestDefaultRegistration::test_default_preparers_register` and `test_default_preparers_count` — they rely on the hardcoded fallback block actually running.

**How to avoid:** PLUG-05 requires tests validate the entry-point path. Add a dedicated `TestEntryPointDiscovery` class testing `importlib.metadata.entry_points(group=...)` directly. The existing "does geo appear in supported_databases" tests remain valid (they test behavior, not mechanism) but the counts must be rechecked with fallback gated.

**Warning signs:** Tests pass with fallback enabled but fail with fallback gated.

### Pitfall 5: MetaboLights is Missing from Queue Preparer Hardcoded Block

**What goes wrong:** The `QueuePreparationService._register_default_preparers()` hardcoded fallback does NOT include MetaboLights (only GEO, PRIDE, SRA, MassIVE). But `DownloadOrchestrator._register_default_services()` DOES include MetaboLights.

**Why it happens:** MetaboLights was added to the download service fallback but the queue preparer fallback was not updated to match.

**Impact for Phase 5:** MetaboLights queue preparer currently has NO registration path (neither entry-point nor hardcoded fallback). After Phase 5, it will be registered only via entry point. This is correct behavior, but the asymmetry means PLUG-04 ("all 5 databases discoverable via entry points") is the ONLY registration path for MetaboLights queue preparer — making PLUG-03 (declare before gating) especially critical.

**How to avoid:** Add `metabolights` to the entry-point declarations first. Verify MetaboLights queue preparation works before gating fallback.

---

## Code Examples

Verified patterns from existing codebase:

### Current Hardcoded Fallback Structure (QueuePreparationService)

```python
# Source: lobster/services/data_access/queue_preparation_service.py:103-173
def _register_default_preparers(self) -> None:
    # Phase 1: Discover queue preparers from entry points
    try:
        from lobster.core.component_registry import component_registry
        for name, preparer_cls in component_registry.list_queue_preparers().items():
            try:
                preparer = preparer_cls(self.data_manager)
                self.register_preparer(preparer)
            except Exception as e:
                logger.warning(f"Failed to load queue preparer '{name}' from entry point: {e}")
    except Exception as e:
        logger.debug(f"Entry point queue preparer discovery skipped: {e}")

    # Phase 2: Hardcoded fallback preparers (GEO, PRIDE, SRA, MassIVE)
    try:
        from lobster.services.data_access.geo_queue_preparer import GEOQueuePreparer
        self.register_preparer(GEOQueuePreparer(self.data_manager))
    except ImportError as e:
        logger.debug(f"GEOQueuePreparer not available: {e}")
    # ... PRIDE, SRA, MassIVE ...
    # NOTE: MetaboLights is NOT in this fallback block
```

### Existing Entry-Point Declaration Pattern

```toml
# Source: pyproject.toml:269-274 — the existing lobster.services pattern to mirror

[project.entry-points."lobster.services"]
publication_processing = "lobster.services.orchestration.publication_processing_service:PublicationProcessingService"
publication_queue = "lobster.core.publication_queue:PublicationQueue"
```

### ComponentRegistry Discovery (already works)

```python
# Source: lobster/core/component_registry.py:317-323
# These are already present — no changes needed to ComponentRegistry

self._load_entry_point_group("lobster.download_services", self._download_services)
self._load_entry_point_group("lobster.queue_preparers", self._queue_preparers)
```

### Runtime Verification Command

```bash
# After uv pip install -e ., verify entry points are registered:
python -c "
from lobster.core.component_registry import component_registry
component_registry.reset()  # force fresh load
component_registry.load_components()
print('Queue preparers:', list(component_registry.list_queue_preparers().keys()))
print('Download services:', list(component_registry.list_download_services().keys()))
"
# Expected output:
# Queue preparers: ['geo', 'sra', 'pride', 'massive', 'metabolights']
# Download services: ['geo', 'sra', 'pride', 'massive', 'metabolights']
```

### Mock Entry-Point Discovery in Tests

```python
# Source: tests/unit/core/test_component_registry.py:38-57 — existing mock pattern

@pytest.fixture
def mock_entry_points():
    mock_ep = Mock()
    mock_ep.name = "geo"
    mock_ep.value = "lobster.services.data_access.geo_queue_preparer:GEOQueuePreparer"
    mock_ep.dist = Mock()
    mock_ep.dist.name = "lobster-ai"
    mock_ep.load.return_value = GEOQueuePreparer
    return {"lobster.queue_preparers": [mock_ep]}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `pkg_resources.iter_entry_points()` | `importlib.metadata.entry_points(group=...)` | Python 3.9+ stdlib | No runtime dep on setuptools |
| Hardcoded service registry | Entry-point based discovery | Phase 5 target | External packages can add databases |

**Current state (pre-Phase-5):**
- `ComponentRegistry` already discovers `lobster.queue_preparers` and `lobster.download_services` at load time (lines 317-323)
- Both routers attempt entry-point discovery first (Phase 1 of `_register_default_*`) but get zero results because no entry points are declared
- Both routers then fall through to hardcoded fallback (Phase 2) which populates the registry
- The hardcoded fallback has no gate — it always runs

**Target state (post-Phase-5):**
- `pyproject.toml` declares all 5 queue preparers and 5 download services as entry points
- Both routers' Phase 1 succeeds, populating the registry from entry points
- Phase 2 (hardcoded fallback) is gated by `_ALLOW_HARDCODED_FALLBACK = False` and never runs in normal operation
- External packages can declare their own `lobster.queue_preparers` entry points with zero core changes

---

## Open Questions

1. **MetaboLights Queue Preparer instantiation signature**
   - What we know: `MetaboLightsQueuePreparer` inherits `IQueuePreparer.__init__(self, data_manager)` — standard signature
   - What's unclear: Whether it has additional `__init__` params that would prevent the `preparer_cls(self.data_manager)` pattern used in the registry loading loop
   - Recommendation: Read `metabolights_queue_preparer.py` constructor before writing the entry-point declaration. If no custom `__init__`, the class is a direct entry-point target (not a factory).

2. **`_ALLOW_HARDCODED_FALLBACK` flag location**
   - What we know: Two files need the flag (`queue_preparation_service.py` and `download_orchestrator.py`)
   - What's unclear: Whether a single shared constant in `component_registry.py` or a separate flag per file is cleaner
   - Recommendation: Module-level constant per file (`_ALLOW_HARDCODED_FALLBACK = False`) is simpler and avoids cross-module coupling. The two files are distinct concerns.

3. **`uv pip install -e .` vs `make dev-install` in CI**
   - What we know: The pyproject.toml uses setuptools as build backend; entry points are written to `*.dist-info/entry_points.txt` during install
   - What's unclear: Whether CI/CD runs `make dev-install` or direct `uv pip install -e .` before tests
   - Recommendation: Add a verification step in the commit checklist to confirm entry points appear after install.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 7.0+ with pytest-mock |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/unit/services/data_access/test_queue_preparation_service.py tests/unit/services/data_access/test_download_services.py -x -q` |
| Full suite command | `pytest tests/unit && pytest tests/integration` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PLUG-01 | Queue preparers discovered via entry points | unit | `pytest tests/unit/services/data_access/test_queue_preparation_service.py -k "entry_point" -x` | Wave 0 |
| PLUG-02 | Download services discovered via entry points | unit | `pytest tests/unit/services/data_access/test_download_services.py -k "entry_point" -x` | Wave 0 |
| PLUG-03 | Entry-point declarations exist before fallback gated | contract | `pytest tests/unit/services/data_access/test_plugin_registration.py::test_all_5_databases_discoverable -x` | Wave 0 |
| PLUG-04 | All 5 databases discoverable | contract | `pytest tests/unit/services/data_access/test_plugin_registration.py -x` | Wave 0 |
| PLUG-05 | Updated tests validate EP path | unit | `pytest tests/unit/services/data_access/ -x -q` | Existing (modify) |
| PLUG-06 | Hardcoded fallback gated | unit | `pytest tests/unit/services/data_access/test_queue_preparation_service.py::TestFallbackGating -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/unit/services/data_access/ -x -q`
- **Per wave merge:** `pytest tests/unit && pytest tests/integration`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/unit/services/data_access/test_plugin_registration.py` — covers PLUG-03, PLUG-04 (new contract tests asserting entry-point declarations)
- [ ] New `TestEntryPointDiscovery` class inside `test_queue_preparation_service.py` — covers PLUG-01
- [ ] New `TestEntryPointDiscovery` class inside `test_download_services.py` — covers PLUG-02
- [ ] New `TestFallbackGating` class inside `test_queue_preparation_service.py` — covers PLUG-06 (asserts `_ALLOW_HARDCODED_FALLBACK = False` skips hardcoded block)

---

## Sources

### Primary (HIGH confidence)

- Codebase inspection — `lobster/core/component_registry.py` (entry-point groups, discovery loop, ComponentRegistry API)
- Codebase inspection — `lobster/services/data_access/queue_preparation_service.py` (current two-phase registration, Phase 1 entry-point path, Phase 2 hardcoded fallback)
- Codebase inspection — `lobster/tools/download_orchestrator.py` (same two-phase pattern for download services)
- Codebase inspection — `pyproject.toml` (build backend, existing entry-point sections, absence of queue_preparers/download_services declarations)
- Codebase inspection — `tests/unit/services/data_access/test_queue_preparation_service.py` and `test_download_services.py` (test assertions that need updating)
- Codebase inspection — `kevin_notes/UNIFIED_CLEANUP_PLAN.md` (PR-5 work items and amendments A3, A4)
- Runtime check — `entry_points(group='lobster.queue_preparers')` returns `[]` (confirmed no declarations exist)

### Secondary (MEDIUM confidence)

- Python packaging documentation — entry-point discovery behavior with `importlib.metadata` and setuptools editable installs

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all tooling is pre-existing and already in use
- Architecture: HIGH — the two-phase pattern is already coded; Phase 5 is wiring + gating
- Pitfalls: HIGH — confirmed by reading the actual source files and runtime verification

**Research date:** 2026-03-04
**Valid until:** 2026-05-04 (30 days; pyproject.toml entry-point semantics are stable)
