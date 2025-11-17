# Test Archive

This directory contains **deprecated test files** from previous implementation phases. These tests have been archived (not deleted) to preserve git history and provide context for the architectural evolution of the research agent refactoring project.

## Archived Tests

### 1. `test_publication_resolver.py`
- **Date Archived**: 2025-01-13
- **Original Location**: `tests/unit/test_publication_resolver.py`
- **Why Archived**: Tests legacy `PublicationResolver` class from Phase 0 (pre-refactor)
- **Replaced By**:
  - Phase 1: Provider architecture (`tests/unit/tools/providers/test_provider_registry.py`)
  - Phase 2: `ContentAccessService` with three-tier cascade (`tests/unit/tools/test_content_access_service.py`)
- **Git History**: Use `git log --follow tests/archive/test_publication_resolver.py` to view full history

**Architectural Changes:**
- Phase 0 (Archived): Monolithic `PublicationResolver` class handled all publication identification and resolution
- Phase 1 (Current): Modular provider architecture with capability-based routing:
  - `AbstractProvider` (fast abstract retrieval via NCBI)
  - `PubMedProvider` (PMC XML, structured content)
  - `WebpageProvider` (publisher webpage extraction)
  - `PDFProvider` (Docling-based PDF parsing)
  - `ProviderRegistry` (auto-detection and routing)

**Test Coverage Migration:**
- Identifier parsing → `test_provider_registry.py::TestProviderRegistry::test_identifier_parsing`
- PMID/DOI resolution → `test_content_access_service.py::TestContentAccessService::test_extract_metadata`
- PMC access → Real API test in `test_provider_registry.py::TestPubMedProvider::test_get_pmc_fulltext_real`

---

### 2. `test_unified_content_service.py`
- **Date Archived**: 2025-01-13
- **Original Location**: `tests/unit/tools/test_unified_content_service.py`
- **Why Archived**: Tests legacy `UnifiedContentService` class from Phase 1 (mid-refactor)
- **Replaced By**:
  - Phase 2: `ContentAccessService` with provider architecture integration (`tests/unit/tools/test_content_access_service.py`)
  - Enhanced three-tier cascade strategy (PMC → Webpage → PDF)
- **Git History**: Use `git log --follow tests/archive/test_unified_content_service.py` to view full history

**Architectural Changes:**
- Phase 1 (Archived): `UnifiedContentService` with two-tier strategy (quick abstract vs full content)
- Phase 2 (Current): `ContentAccessService` with three-tier cascade and provider delegation:
  - **Tier 1 (Fast)**: `AbstractProvider` for quick abstract retrieval (200-500ms, NCBI)
  - **Tier 2 (Structured)**: `PubMedProvider` for PMC XML full text (500ms, 95% accuracy)
  - **Tier 3 (Fallback)**: `WebpageProvider` → `PDFProvider` cascade (2-8 seconds)

**Test Coverage Migration:**
- `get_quick_abstract()` → `test_content_access_service.py::TestContentAccessService::test_tier1_abstract_retrieval_real`
- `get_full_content()` with webpage-first → `test_content_access_service.py::TestContentAccessService::test_tier2_pmc_cascade_real`
- `extract_methods_section()` → `test_content_access_service.py::TestContentAccessService::test_methods_extraction_real`
- Cache integration → Integrated with `DataManagerV2` provenance tracking

**Why Not Adapted (Instead of Archived):**
- Service name changed (`UnifiedContentService` → `ContentAccessService`)
- Method signatures changed (delegation to provider architecture)
- Initialization changed (requires `ProviderRegistry`, not direct provider instantiation)
- Completely new three-tier cascade strategy (PMC-first vs webpage-first)
- Integration with Phase 2 provider architecture makes adaptation more complex than rewriting

---

## Archive Policy

**When to Archive Tests:**
1. Service/class renamed or refactored beyond recognition
2. Architectural pattern fundamentally changed (e.g., monolithic → modular)
3. Test would require >60% rewrite to adapt to new architecture
4. New tests with real API calls replace mock-based legacy tests

**When to Adapt Tests (Not Archive):**
1. Minor method signature changes
2. Parameter additions with backward compatibility
3. Bug fixes or enhancements to existing functionality
4. Coverage improvements for existing test subjects

**Git History Preservation:**
- Tests are **moved** (not deleted) to preserve `git log --follow` functionality
- Original file paths documented in this README
- Commit dates and authors preserved in git history
- Use `git log --all -- tests/archive/<filename>` to see full history including pre-archive commits

---

## Running Archived Tests

**⚠️ These tests will NOT run successfully in the current codebase.** They reference deleted classes and modules.

If you need to run archived tests (e.g., for git bisect or historical debugging):

```bash
# Checkout specific commit when test was valid
git checkout <commit-hash>

# Run the specific test file
pytest tests/unit/test_publication_resolver.py -v

# Return to current branch
git checkout dev_provenance_271025
```

---

## Replacement Test Files (Phase 6)

**✅ New test files with real API integration:**

| Legacy Test | Replacement Test | Coverage | Real API Tests |
|-------------|------------------|----------|----------------|
| `test_publication_resolver.py` | `tests/unit/tools/providers/test_provider_registry.py` | 95%+ | 4 tests (PMID, DOI, PMC, GEO) |
| `test_unified_content_service.py` | `tests/unit/tools/test_content_access_service.py` | 90%+ | 10 tests (3-tier cascade, methods) |

**🔬 Real API Test Strategy (Phase 6):**
- **pytest markers**: `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.real_api`
- **Rate limiting**: 0.5-1 second sleeps between consecutive API calls
- **CI/CD integration**: Separate GitHub Actions job for integration tests
- **Coverage maintenance**: Hybrid approach (unit + integration) maintains ≥85% total coverage

---

## API Keys Required for Replacement Tests

**Environment Variables (tests/README.md for full details):**

```bash
# Required for real API integration tests
export NCBI_API_KEY="your_ncbi_api_key"              # PubMed, PMC access (0.33/sec → 10/sec with key)
export AWS_BEDROCK_ACCESS_KEY="your_access_key"      # LLM for methods extraction
export AWS_BEDROCK_SECRET_ACCESS_KEY="your_secret"   # LLM for methods extraction

# Optional (tests work without, but slower)
export ANTHROPIC_API_KEY="your_anthropic_key"       # Alternative LLM provider
```

**How to Run Real API Tests:**

```bash
# Run all integration tests with real API calls
pytest tests/unit/tools/providers/ tests/unit/tools/test_content_access_service.py -m "integration" -v

# Run only fast unit tests (no API calls)
pytest tests/unit/tools/ -m "not integration and not slow" -v

# Run with coverage (hybrid unit + integration)
pytest tests/unit/tools/ --cov=lobster/tools --cov-report=term-missing --cov-report=html
```

---

## Phase History & Documentation

**Complete refactoring timeline:**

| Phase | Description | Implementation | Tests |
|-------|-------------|----------------|-------|
| **Phase 0** | Legacy monolithic architecture | `PublicationResolver` | Archived: `test_publication_resolver.py` |
| **Phase 1** | Provider architecture introduction | `AbstractProvider`, `PubMedProvider`, `WebpageProvider`, `PDFProvider`, `ProviderRegistry` | New: `test_provider_registry.py` |
| **Phase 2** | Service consolidation & three-tier cascade | `ContentAccessService`, `MetadataValidationService` | New: `test_content_access_service.py` |
| **Phase 3** | Agent separation & metadata operations | `research_agent`, `metadata_assistant` split | New: `test_metadata_assistant.py`, updated: `test_research_agent.py` |
| **Phase 4** | Documentation & workflow examples | Handoff protocols, workflow patterns in agent system prompts | N/A (documentation phase) |
| **Phase 5** | Documentation expansion | Comprehensive examples, decision thresholds, report templates | N/A (documentation phase) |
| **Phase 6** | **Test suite migration (current)** | Real API integration tests, archive legacy tests | This README + new test files |

**Documentation:**
- Full phase summaries: `kevin_notes/publisher/phase_1_summary.md` through `phase_5_summary.md`
- Phase 6 plan: `kevin_notes/publisher/phase_6_plan.md`
- Phase 6 validation checklist: `kevin_notes/publisher/phase_6_validation_checklist.md`

---

## Questions or Issues?

If you encounter issues with:
- **Archived tests**: Check git history (`git log --follow`) to understand when/why archived
- **Replacement tests**: See `tests/README.md` for API key setup and execution instructions
- **Phase documentation**: Check `kevin_notes/publisher/` for complete phase summaries and plans
- **Integration tests failing**: Verify API keys set, check rate limiting (0.5-1s between calls)

**Contact:** See `CLAUDE.md` for development guidelines and architectural context.

---

## Recent Infrastructure Changes (2025-11-16)

### Documentation Relocation
Pre-release testing documentation has been moved to a more appropriate location:
- **Old Location**: `tests/pre-release-report/` (18 files)
- **New Location**: `docs/testing/pre-release-2025-09/` (15 markdown files, 2 Python scripts, 1 JSON file)
- **Rationale**: Test reports and analysis scripts belong in documentation, not in test directory

### Services Directory Removal
The deprecated `tests/services/` directory has been removed:
- **Reason**: Tests duplicated functionality in `tests/unit/tools/` and used outdated service patterns
- **Migration**: All service tests consolidated into `tests/unit/tools/` with proper organization
- **Date**: 2025-11-16

### Fixture Documentation Enhancement
All fixtures in `tests/conftest.py` have been enhanced with:
- Comprehensive docstrings explaining purpose and usage
- Type hints for all fixtures
- Example code showing proper fixture usage
- Fixture dependency map in module docstring (26 documented fixtures total)
