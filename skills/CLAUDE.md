# CLAUDE.md — Skill Evaluation & Development

Working directory for Lobster AI skill authoring, testing, and evaluation. USE THIS CLAUDE.md FILE (/Users/tyo/Omics-OS/lobster/skills/CLAUDE.md) as a scratchpad during your work!

---

## What Lives Here

| Path | Purpose |
|------|---------|
| `lobster-dev/` | Developer skill — teaches coding agents to build Lobster plugins |
| `lobster-use/` | End-user skill — teaches coding agents to USE Lobster for analysis |
| `../.testing/skill-eval/` | Docker-based eval harness (images, prompts, ground truth, results) |
| `../packages/` | Real agent packages (reference implementations for skill accuracy) |
| `../lobster/scaffold/` | Built-in scaffold generator (`lobster scaffold agent`) — replaces old Copier template |

---

## Main objective of skill

The lobster-dev skill should be able to create ANY possible extension to lobster to solve ANY problem a user might have in the biological domain. This means from simple database integrations to visualizations to compolex integration of omics-data. This skill follows the newly developed AQUADIF principle for new agents (but not every user request requires creating new agents). **AQUADIF** is the 10-category taxonomy for Lobster AI tools. Every tool declares what it does (category) and whether it must produce provenance — making the system introspectable, enforceable, and teachable to coding agents. (./lobster-dev/references/aquadif-contract.md)

## Skill Eval Infrastructure

### How It Works

Three coding agents (Claude, Gemini, Codex) are given the same prompt ("build a lobster-<domain or problem> agent") inside Docker containers. **WITH** condition loads the skill; **WITHOUT** is the control. We evaluate generated code against ground truth and the Lobster plugin contract.

### Running Tests

```bash
cd ../.testing/skill-eval

# Build all 7 images (base + 3 agents x 2 conditions)
./build.sh

# Run a trial
./run-test.sh --agent claude --condition with --domain epigenomics --campaign <name> --trial 1
./run-test.sh --agent gemini --condition with --domain epigenomics --campaign <name> --trial 1
./run-test.sh --agent codex  --condition with --domain epigenomics --campaign <name> --trial 1

# Interactive debugging (drops into container shell)
./run-test.sh --agent claude --condition with --interactive
```

### API Keys

| Agent | Key Source | Notes |
|-------|-----------|-------|
| Claude | `ANTHROPIC_API_KEY` env or AWS Bedrock creds | Bedrock configured in image |
| Gemini | `GOOGLE_API_KEY` from `~/.env` or env var | Passed via `-e` to Docker |
| Codex | `~/.codex/auth.json` mounted into container | `OPENAI_API_KEY` env var alone does NOT work — Codex CLI reads from auth.json |

### Key Files

| File | Purpose |
|------|---------|
| `.testing/skill-eval/run-test.sh` | Test runner — builds prompt, launches container, captures results |
| `.testing/skill-eval/build.sh` | Image builder — concatenates SKILL.md + references into AGENT.md/GEMINI.md |
| `.testing/skill-eval/ground-truth/epigenomics.json` | Expected tool patterns, categories, counts |
| `.testing/skill-eval/prompts/epigenomics.txt` | Domain prompt given to all agents |
| `.testing/skill-eval/Dockerfile.{claude,gemini,codex}` | Per-agent Docker images |

### Results Structure

```
.testing/skill-eval/results/
├── smoke-01/                          # First campaign (pre-skill-fix)
│   └── claude-with-epigenomics-t1/
│       ├── metadata.json              # Duration, exit code, timestamps
│       ├── output.log                 # Agent's final summary
│       ├── workspace/packages/        # Generated code
│       └── evaluation.md              # Human-written evaluation
├── iter-02/                           # Second campaign (post-skill-fix)
│   ├── gemini-with-epigenomics-t1/
│   ├── codex-with-epigenomics-t1/
│   └── comparison.md                  # Cross-agent comparison
```

### Evaluation Dimensions (6)

1. **Structure Audit** — AGENT_CONFIG position, entry points, PEP 420, factory signature
2. **Weaknesses Found** — Anti-patterns, wrong paths, missing files
3. **Gaps** — Missing capabilities vs ground truth patterns (9 required)
4. **Good Practices** — AQUADIF metadata, provenance, service contracts
5. **Skill Change Recommendations** — What to fix in the skill
6. **Synthesis** — Grade (A-F), agent personality profile

---

## Current State

### Completed Evaluations

| Campaign | Agent | Grade | Tools | Duration | Key Finding |
|----------|-------|-------|-------|----------|-------------|
| smoke-01 | Claude Opus 4.6 | **B+** | 13 | 689s | Real implementations (TMM, BMIQ, BH). `__init__.py` fail (pre-fix). |
| iter-02 | Gemini 3.1 Pro | **C** | 8 | 208s | Correct skeleton, all stubs. Wrong import paths. Module-level imports. |
| iter-02 | Codex gpt-5.3 | **B-** | 15 | 242s | Best AQUADIF understanding (9/10 categories). Built standalone, not plugin. |

**Note:** smoke-01 and iter-02 were run BEFORE `lobster scaffold` and `lobster validate-plugin` existed. iter-03 will be the first evaluation with these features available to agents.

### Skill Fixes Applied (Pre iter-03)

| Change | File | What |
|--------|------|------|
| CHANGE 1 (Critical) | `creating-agents.md` | Replaced `try/except ImportError` with PEP 420 pattern |
| CHANGE 2 (Medium) | `aquadif-contract.md` | Added signal track / fragment file IMPORT examples |
| CHANGE 3 (High) | Both files | Added "Data Loading Boundary: data_expert vs Domain IMPORT Tools" section |
| CHANGE 4 (Major) | `scaffold.md` + `SKILL.md` | Added `lobster scaffold agent` as mandatory first step |
| CHANGE 5 (Major) | `scaffold.md` + `SKILL.md` | Added `lobster validate-plugin` with 7 structural checks |

### What scaffold + validate-plugin Now Enforce

These structural issues from iter-02 are now **automatically prevented** by the scaffold:
- `src/` layout → scaffold generates flat layout (`lobster/agents/{domain}/`)
- Missing pyproject.toml → scaffold generates it with correct entry points
- Missing AGENT_CONFIG → scaffold generates it at module top
- Wrong `__init__.py` → scaffold generates PEP 420 compliant files
- Missing contract tests → scaffold generates AQUADIF contract test template

And `validate-plugin` catches anything that slips through (7 checks: pyproject.toml, entry points, AGENT_CONFIG, PEP 420, factory function, service 3-tuple, AQUADIF metadata).

### Infrastructure Changes (for iter-03)

- Docker images now install **lobster-ai from source** (not PyPI) — `build.sh` rsyncs lobster source into build context, Dockerfiles install from `/tmp/lobster-src`. Agents get `lobster scaffold` + `lobster validate-plugin` before PyPI release.
- `run-test.sh`: Added `~/.codex/auth.json` mount for Codex Docker auth.

---

## Roadmap

### Main Objective

Make the lobster-dev skill reliably teach ANY coding agent (Claude, Gemini, Codex) to produce a **structurally correct, ecosystem-integrated** Lobster plugin package. Target: all agents score B+ or higher.

### Phase 1: Fix Known Skill Gaps — MOSTLY COMPLETE

Skill changes from smoke-01 + iter-02 evaluations:

- [x] Fix `__init__.py` code example (CHANGE 1)
- [x] Broaden IMPORT examples (CHANGE 2)
- [x] Add data_expert boundary (CHANGE 3)
- [x] `lobster scaffold` as mandatory first step — enforces flat layout, pyproject.toml, AGENT_CONFIG, PEP 420
- [x] `lobster validate-plugin` — 7 structural checks catch remaining issues

Remaining nice-to-haves (lower priority now that scaffold enforces these):
- [ ] Add import path reference table to creating-agents.md (helps Gemini avoid hallucinated paths)
- [ ] Add visual lazy import annotation (module-level vs factory-level separation)

### Phase 2: Template Integration — COMPLETED

`lobster scaffold agent` generates AQUADIF-compliant packages. `lobster validate-plugin` validates them.
Old Copier template deleted. All scaffolding built into core at `lobster/scaffold/`.

### Phase 3: iter-03 — Re-run All 3 Agents (NEXT)

Re-run Claude + Gemini + Codex with current skill (scaffold + validate-plugin available).
This is the first evaluation where agents have access to `lobster scaffold` and `lobster validate-plugin`.
Expect significant grade improvements from structural enforcement.

### Phase 4: WITHOUT Condition Baseline

Run all 3 agents WITHOUT skills to measure skill impact delta.

### Phase 5: Automated Scoring

Parse generated files programmatically (AST checks, grep patterns, structured JSON output).

### Phase 6: Multi-Domain

Expand beyond epigenomics (`prompts/multi-omics.txt` already exists).

---

## Scratchpad

### Known Issues

1. **Gemini hallucinated import paths** — `lobster.core.config` and `lobster.core.client` don't exist. Skill shows correct paths but Gemini didn't follow. An import path reference table in creating-agents.md might help. **Will re-evaluate after iter-03** — scaffold may make this less of an issue since generated code has correct imports.

2. **Codex exit code 1** — "Failed to shutdown rollout recorder" — Codex CLI internal cleanup error, code gen succeeds. `run-test.sh` should check workspace files exist before treating non-zero exit as failure.

3. **AQUADIF metadata migration needed** — 5/9 existing packages fail AQUADIF checks (separate from eval work, tracked in main lobster backlog).

### Todos

- [ ] **Run iter-03** — all 3 agents WITH skill, epigenomics domain (NEXT)
- [ ] Run WITHOUT condition baseline for all 3 agents
- [ ] Add import path reference table to creating-agents.md (if Gemini still hallucinates in iter-03)
- [ ] Handle Codex "rollout recorder" exit code in run-test.sh
- [ ] Migrate AQUADIF metadata to existing packages (metabolomics, transcriptomics, proteomics, ml)

### Agent Personality Notes (from smoke-01 / iter-02)

- **Claude** — "The Integrator". Real algorithms, follows patterns precisely. Slow (689s). Over-implements.
- **Gemini** — "The Scaffolder". Fast (208s), correct structure, zero depth. Hallucination risk.
- **Codex** — "The Engineer". Best AQUADIF grasp, DRY patterns. Builds standalone instead of plugins.

**iter-03 hypothesis:** With `lobster scaffold` available, Codex's standalone-build problem should disappear and Gemini's structural issues should be enforced away. Implementation depth remains a model capability question.

### Open Questions

1. **Should we test lobster-use skill too?** Currently only evaluating lobster-dev.
2. **Service implementation depth** — Claude writes real algorithms; Gemini writes stubs. Model knowledge, not skill problem.
