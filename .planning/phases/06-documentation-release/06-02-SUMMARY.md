---
phase: 06-documentation-release
plan: "02"
subsystem: docs-site, skills
tags: [aquadif, documentation, docs-site, skills, monitoring]
dependency_graph:
  requires: [05-02]
  provides: [DOC-03, DOC-04, DOC-05]
  affects: [docs-site/extending, skills/lobster-dev, skills/lobster-use]
tech_stack:
  added: []
  patterns: [MDX page with Fumadocs components, skill reference addendum]
key_files:
  created:
    - docs-site/content/docs/extending/aquadif-taxonomy.mdx
  modified:
    - docs-site/content/docs/extending/meta.json
    - skills/lobster-dev/references/aquadif-contract.md
    - skills/lobster-dev/references/testing.md
    - skills/lobster-use/SKILL.md
decisions:
  - "aquadif-taxonomy.mdx placed last in extending/ pages array — it is a reference page, not a getting-started page"
  - "Runtime Monitoring section appended before Migration appendix in aquadif-contract.md — preserves existing content order"
  - "lobster-use AQUADIF mention is one paragraph only — end users need category names in context, not full contract details"
metrics:
  duration: 206s
  completed: "2026-03-01"
  tasks: 2
  files: 5
---

# Phase 6 Plan 02: AQUADIF Documentation — Docs-Site Page + Skill Updates Summary

**One-liner:** Public AQUADIF taxonomy page on docs-site + monitoring section in skill reference + pytest -m contract command in testing guide + AQUADIF mention for end-user skill.

## What Was Built

### Task 1: docs-site AQUADIF taxonomy page and navigation (DOC-03)

Created `docs-site/content/docs/extending/aquadif-taxonomy.mdx` — a public-facing documentation page for external plugin developers. Content sections:

1. **Quick Reference table** — all 10 categories with definitions, provenance requirements, and example tools
2. **Metadata Assignment Pattern** — `@tool` + `.metadata` + `.tags` pattern with complete code example; 3 key rules
3. **Provenance Rules** — which tools require provenance, UTILITY exception, hollow provenance (`ir=None`) bridge pattern
4. **Contract Testing** — `AgentContractTestMixin` table of 14 validated properties, `pytest -m contract` commands, link to Testing page
5. **Runtime Monitoring** — what `AquadifMonitor` tracks (4 metrics), how it's wired (4 steps), plugin author guidance
6. **Category Decision Guide** — 80% rule, FILTER/PREPROCESS/QUALITY/ANALYZE/ANNOTATE boundary cases, quick decision table
7. **NextSteps component** — links to Testing, Plugin Contract, and Package Structure

Updated `docs-site/content/docs/extending/meta.json` — added `"aquadif-taxonomy"` to pages array (last position, after `"testing"`).

**Verification:** `npm run build` succeeded with zero errors.

### Task 2: Skill reference updates (DOC-04, DOC-05)

**A. aquadif-contract.md — Runtime Monitoring section**

Added new "## Runtime Monitoring" section before the "## Migrating Existing Agents" appendix. Documents:
- 4 tracked metrics: category distribution, provenance status, CODE_EXEC log, session summary
- 4-step wiring chain: client.py → graph.py → TokenTrackingCallback → DataManagerV2
- Design properties: pure stdlib, fail-open, thread-safe, bounded
- Plugin author guidance: zero configuration required

**B. testing.md — AQUADIF Contract Tests section**

Added new "## AQUADIF Contract Tests" section before "## Best Practices". Documents:
- `pytest -m contract` for all agents
- `pytest -m contract -k transcriptomics` for single package
- Cross-reference to `aquadif-contract.md` for detailed contract method descriptions

**C. lobster-use SKILL.md — AQUADIF mention in Agent System section**

Added one paragraph after the agent table (before "Details and hierarchy"). Mentions all 10 category names in context, explains the system routes requests automatically. Keeps it lightweight — end users don't need contract details.

## Commits

| Task | Commit | Files |
|------|--------|-------|
| Task 1 (docs-site repo) | 7b407ff | aquadif-taxonomy.mdx, meta.json |
| Task 2 (lobster repo) | b7b711f | aquadif-contract.md, testing.md, SKILL.md |

## Verification Results

All 6 plan verification criteria confirmed passing:

1. `aquadif-taxonomy.mdx` exists in `docs-site/content/docs/extending/`
2. `meta.json` pages array includes `"aquadif-taxonomy"`
3. `npm run build` succeeded (zero errors, full static build)
4. `AquadifMonitor` found in Runtime Monitoring section of `aquadif-contract.md`
5. `pytest -m contract` explicitly shown in `testing.md`
6. `AQUADIF` mention found in Agent System section of `lobster-use/SKILL.md`

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `docs-site/content/docs/extending/aquadif-taxonomy.mdx` — FOUND
- `docs-site/content/docs/extending/meta.json` (contains aquadif-taxonomy) — FOUND
- `skills/lobster-dev/references/aquadif-contract.md` (contains AquadifMonitor) — FOUND
- `skills/lobster-dev/references/testing.md` (contains pytest -m contract) — FOUND
- `skills/lobster-use/SKILL.md` (contains AQUADIF) — FOUND
- Commits 7b407ff, b7b711f — FOUND
