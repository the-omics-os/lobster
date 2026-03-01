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

### Skill Fixes Applied (This Iteration)

| Change | File | What |
|--------|------|------|
| CHANGE 1 (Critical) | `lobster-dev/references/creating-agents.md` | Replaced `try/except ImportError` code example with PEP 420 pattern in `__init__.py` section |
| CHANGE 2 (Medium) | `lobster-dev/references/aquadif-contract.md` | Added signal track / fragment file IMPORT examples |
| CHANGE 3 (High) | Both files | Added "Data Loading Boundary: data_expert vs Domain IMPORT Tools" section |

### Infrastructure Fixes

- `run-test.sh`: Added `~/.codex/auth.json` mount for Codex Docker auth

---

## Roadmap

### Main Objective

Make the lobster-dev skill reliably teach ANY coding agent (Claude, Gemini, Codex) to produce a **structurally correct, ecosystem-integrated** Lobster plugin package. Target: all agents score B+ or higher.

### Phase 1: Fix Known Skill Gaps (current)

Skill changes identified from smoke-01 + iter-02 evaluations:

- [x] Fix `__init__.py` code example (CHANGE 1)
- [x] Broaden IMPORT examples (CHANGE 2)
- [x] Add data_expert boundary (CHANGE 3)
- [ ] Add visual lazy import annotation — separate module-level (AGENT_CONFIG only) from factory-level (everything else)
- [ ] Ban `src/` layout explicitly — "MUST use flat layout, NOT src/"
- [ ] Add "you are a plugin, not standalone" note — anndata.AnnData required, no custom data containers
- [ ] Emphasize pyproject.toml as mandatory first file
- [ ] Document correct import paths in reference table (not just in template)
- [ ] Fix factory return type emphasis — "MUST return create_react_agent() CompiledGraph"

### Phase 2: Template Integration — COMPLETED

**Resolved:** `lobster scaffold agent` now generates AQUADIF-compliant plugin packages.
The old Copier template (`lobster-agent-template/`) has been deleted. All scaffolding
is now built into the core package at `lobster/scaffold/`.

### Phase 3: WITHOUT Condition Baseline

Run all 3 agents WITHOUT skills to measure skill impact. This gives us:
- Delta between with/without per agent
- Evidence that skills actually improve output (not just agent capability)

### Phase 4: Automated Scoring

Parse generated files programmatically:
- AST-check for AGENT_CONFIG position
- Grep for entry points in pyproject.toml
- Count tools and verify AQUADIF metadata
- Check provenance `ir=ir` calls
- Score against ground truth patterns
- Output structured JSON for comparison

### Phase 5: Multi-Domain

Expand beyond epigenomics:
- `prompts/multi-omics.txt` already exists
- Add more ground truth files
- Validate skill generalization across domains

---

## Scratchpad

### Bugs

1. ~~**Template `__init__.py` eager imports**~~ — FIXED: Old Copier template deleted. `lobster scaffold` generates correct PEP 420 pattern.

2. ~~**Template factory has module-level heavy imports**~~ — FIXED: `lobster scaffold` generates correct lazy import pattern.

3. ~~**Template uses `state_modifier` not `prompt`**~~ — FIXED: `lobster scaffold` uses `prompt=system_prompt`.

4. **Gemini hallucinated import paths** — `lobster.core.config` and `lobster.core.client` don't exist. Skill shows correct paths but Gemini didn't follow. Need paths in a more prominent location (reference table?).

5. **Codex exit code 1** — "Failed to shutdown rollout recorder" — Codex CLI internal cleanup error. Code gen succeeded. May need to handle non-zero exit codes that aren't real failures in `run-test.sh`.

6. **AQUADIF metadata migration needed** — Validator found 5/9 existing packages fail AQUADIF checks:
   - 4 packages (metabolomics, transcriptomics, proteomics, ml) have 0 AQUADIF metadata on ~41 tools
   - 1 package (research) has cross-agent import boundary violation
   - 4 packages pass (genomics, visualization, metadata, structural-viz)

### Todos

- [ ] Run iter-03 after applying remaining Phase 1 fixes
- [ ] Run WITHOUT condition for all 3 agents (baseline measurement)
- [x] ~~Update template `__init__.py` to PEP 420 pattern~~ — Replaced with `lobster scaffold`
- [x] ~~Add AQUADIF contract test template to copier template~~ — Built into `lobster scaffold`
- [x] ~~Consider: should `creating-agents.md` tell agents to run copier first?~~ — Skill now routes through `lobster scaffold`
- [ ] Add import path reference table to creating-agents.md
- [ ] Handle Codex "rollout recorder" exit code in run-test.sh (check workspace files exist → treat as success)
- [ ] Migrate AQUADIF metadata to 4 existing packages (metabolomics, transcriptomics, proteomics, ml)
- [ ] Fix cross-agent import in lobster-research (research_agent imports from data_expert)

### Agent Personality Notes (from evaluations)

- **Claude** — "The Integrator". Deep ecosystem knowledge, real algorithms, follows patterns precisely. Slow (689s). Over-implements.
- **Gemini** — "The Scaffolder". Fast (208s), correct structure, zero depth. Hallucination risk on import paths. All stubs.
- **Codex** — "The Engineer". Best AQUADIF grasp, protocol interfaces, DRY patterns. Builds standalone packages instead of plugins. Creates compat shims.

### Open Questions

1. ~~**Template-first vs skill-first?**~~ RESOLVED: `lobster scaffold` generates boilerplate, skill teaches how to fill it in. Agents run scaffold first, then follow skill guidance for domain logic.
2. **Should we test lobster-use skill too?** Currently only evaluating lobster-dev. The end-user skill has different failure modes.
3. **How to handle service implementation depth?** Claude writes real TMM normalization; Gemini writes stubs. Is this a skill problem or a model knowledge problem? Probably model knowledge — can't teach bioinformatics algorithms via a skill.
