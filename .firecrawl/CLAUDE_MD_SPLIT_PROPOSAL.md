# Proposal: Split CLAUDE.md Into Modular Structure

**Goal**: Reduce combined CLAUDE.md context from ~2,049 lines to ~300 lines while preserving all detailed guidance via progressive disclosure.

**Current State**:
- `lobster/CLAUDE.md`: 1,229 lines
- Parent `CLAUDE.md` (omics-os): 820 lines
- **Combined**: ~2,049 lines loaded per session

**Target**: Main file <300 lines, detailed docs loaded on-demand via `@imports`.

---

## Research Synthesis

### Key Findings from 6 Authoritative Sources

| Source | Key Insight |
|--------|-------------|
| **Builder.io Guide** | Use `@imports` syntax, `.claude/rules/` directory, and subdirectory CLAUDE.md files |
| **GitHub citypaul/.dotfiles** | 1,818→300 line reduction plan using modular structure with absolute paths for ~/.claude/ |
| **Anthropic Official Docs** | Keep CLAUDE.md "short and human-readable", use `/init` as starting point, prune ruthlessly |
| **Claude Code Best Practices** | "Ruthlessly prune... if Claude already does something correctly without the instruction, delete it" |
| **HumanLayer Guide** | <60 lines ideal, LLMs follow ~150-200 instructions max, use Progressive Disclosure |
| **Reddit Discussion** | Document references pattern is "game-changer" for large codebases |

### Core Principles Discovered

1. **Less is more**: Frontier LLMs can reliably follow ~150-200 instructions; Claude Code system prompt already uses ~50
2. **Progressive Disclosure**: Don't load everything upfront; point to detailed docs for on-demand loading
3. **Universal applicability**: Only include what's needed in EVERY session
4. **Pointers over copies**: Don't embed code snippets; use `file:line` references
5. **Context is precious**: Every line competes for attention with actual work

### Official `@import` Syntax

```markdown
# In CLAUDE.md
See @docs/architecture.md for detailed architecture
See @README.md for project overview
See @~/.claude/personal-preferences.md for user overrides
```

- **Recursion**: Up to 5 levels of nested imports
- **Absolute paths**: Required for `~/.claude/` files
- **Protection**: Imports inside code blocks are ignored
- **Verification**: Use `/memory` command to verify loaded files

---

## Proposed Structure for Lobster

### Directory Layout

```
lobster/
├── CLAUDE.md                    # Main file (~250 lines) - Core principles + pointers
├── .claude/
│   └── docs/                    # Detailed documentation (loaded on-demand)
│       ├── architecture.md      # Section 2: HOW – Architecture (~150 lines)
│       ├── code-layout.md       # Section 3: WHERE – Code Layout (~250 lines)
│       ├── development-rules.md # Section 4: RULES – Development Guidelines (~450 lines)
│       ├── tooling.md           # Section 5: Tooling, Commands & Environment (~200 lines)
│       └── agent-patterns.md    # Agent-specific patterns & workflows (~150 lines)
```

### Main CLAUDE.md Structure (~250 lines)

```markdown
# CLAUDE.md

System prompt for Lobster AI — professional multi-agent bioinformatics platform.

## Documentation Navigation
- Parent: `../CLAUDE.md` (Monorepo overview)
- Detailed docs: `.claude/docs/` (loaded on-demand)

## 1. Project Overview (inline - essential)
[~30 lines: What Lobster is, core capabilities, design principles]

## 2. Architecture Summary (inline)
**Core principle**: Agent-based system with 4-layer architecture
**Quick reference**:
- CLI → LobsterClientAdapter → AgentClient/CloudLobsterClient → LangGraph → Agents → Services
- DataManagerV2 for multi-modal data orchestration

For detailed architecture diagrams and data flow, see @.claude/docs/architecture.md

## 3. Code Layout Summary (inline)
**Core principle**: PEP 420 namespace package, no `lobster/__init__.py`
**Quick reference**:
- `agents/` - Modular agent folders (config.py, prompts.py, state.py)
- `services/` - Stateless analysis services
- `core/` - Client, data management, provenance
- `tools/` - Utilities & providers

For complete file structure and component reference, see @.claude/docs/code-layout.md

## 4. Hard Rules (inline - non-negotiable)
1. Do NOT edit `pyproject.toml`
2. Prefer editing existing files over adding new ones
3. Use `config/agent_registry.py` for agents
4. Follow modular agent structure (see template)
5. Keep services stateless: pure functions returning 3-tuple
6. Always pass `ir` into `log_tool_usage()`

For complete development rules, patterns, and abstractions, see @.claude/docs/development-rules.md

## 5. Quick Commands (inline)
```bash
make dev-install     # Install with dev deps
make test            # Run all tests
make format          # black + isort
lobster chat         # Interactive mode
lobster query "..."  # Single-turn automation
```

For complete tooling, environment setup, and publishing, see @.claude/docs/tooling.md

## 6. Who You Are – ultrathink (inline)
[Keep this section inline - defines assistant personality]
```

---

## Migration Plan

### Phase 1: Create `.claude/docs/` Structure

1. **Create directory**: `mkdir -p .claude/docs/`

2. **Extract architecture.md**:
   - Section 2.1: 4-Layer Architecture
   - Section 2.2: Client Layer
   - Section 2.3: Data & Control Flow
   - Section 2.4: Download Queue Pattern

3. **Extract code-layout.md**:
   - Section 3.1: Top-Level Structure
   - Section 3.2: Core Components Reference
   - Section 3.3: Agent Roles
   - Section 3.4: Deployment & Infrastructure

4. **Extract development-rules.md**:
   - Section 4.1: Hard Rules (keep summary in main)
   - Section 4.2-4.4: Service/Tool/Provenance patterns
   - Section 4.5: Patterns & Abstractions (all)
   - Section 4.6-4.12: Download architecture, workflows, security, tiering

5. **Extract tooling.md**:
   - Section 5.1: Technology Stack
   - Section 5.2-5.6: Environment, Testing, Running, Publishing, Claude Code integration

6. **Extract agent-patterns.md** (optional):
   - Agent-specific prompts, tools, delegation patterns
   - Referenced when working on agent code

### Phase 2: Update Main CLAUDE.md

1. Replace detailed sections with:
   - Core principle statement (1-2 sentences)
   - Quick reference bullets (3-5 most important points)
   - Import statement: `For details, see @.claude/docs/[file].md`

2. Keep inline:
   - Project Overview (essential context)
   - Hard Rules (non-negotiable, must always be visible)
   - Quick Commands (most frequently used)
   - Who You Are – ultrathink (personality)

### Phase 3: Update Parent CLAUDE.md (omics-os)

Similar reduction:
- Keep: Repository Overview, Business Context summary, Quick Commands
- Extract to `omics-os/.claude/docs/`:
  - `business-context.md` (~200 lines)
  - `sync-automation.md` (~150 lines)
  - `customer-proposals.md` (~250 lines)

### Phase 4: Verification

1. Run `/memory` command to verify imports work
2. Test Claude behavior on common tasks
3. Verify no content lost (compare line counts)
4. Check import recursion depth (<5 levels)

---

## Benefits

### For Context Management
| Before | After |
|--------|-------|
| 2,049 lines loaded every session | ~250 lines loaded every session |
| All detail competes for attention | Details loaded only when relevant |
| Context fills quickly | Context stays lean |

### For Maintenance
- **Focused editing**: Change architecture docs without touching rules
- **Reduced merge conflicts**: Different sections in different files
- **Team ownership**: Different people can own different docs
- **Version control clarity**: Git diffs show which topic changed

### For Claude Performance
- **Better instruction following**: Fewer instructions = higher adherence
- **Relevant context**: Only loads details when working in that area
- **Reduced "forgetting"**: Important rules stay visible in main file

---

## Alternative Approaches Considered

### 1. `.claude/rules/` Directory
- Auto-loads all markdown files in directory
- Con: Still loads everything; less control over what's loaded when

### 2. Subdirectory CLAUDE.md Files
- e.g., `agents/CLAUDE.md`, `services/CLAUDE.md`
- Pro: Auto-loads when working in that directory
- Con: Requires restructuring; may be redundant with imports

### 3. Skills Instead of Docs
- Create skills for specific workflows (e.g., `/add-agent`, `/add-service`)
- Pro: Invoked explicitly; very targeted
- Con: More setup; doesn't cover reference documentation

**Recommendation**: Use `@imports` as primary approach (most flexible, officially supported), with subdirectory CLAUDE.md for highly specialized areas if needed.

---

## Implementation Checklist

- [ ] Create `.claude/docs/` directory
- [ ] Extract architecture.md (~150 lines)
- [ ] Extract code-layout.md (~250 lines)
- [ ] Extract development-rules.md (~450 lines)
- [ ] Extract tooling.md (~200 lines)
- [ ] Update main CLAUDE.md to ~250 lines with imports
- [ ] Verify with `/memory` command
- [ ] Test Claude behavior on representative tasks
- [ ] Apply same pattern to parent CLAUDE.md
- [ ] Update documentation references
- [ ] Commit with descriptive message

---

## References

- [Claude Code Memory Documentation](https://code.claude.com/docs/en/memory)
- [Builder.io CLAUDE.md Guide](https://www.builder.io/blog/claude-md-guide)
- [HumanLayer: Writing a Good CLAUDE.md](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
- [Anthropic Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [GitHub citypaul Split Plan](https://github.com/citypaul/.dotfiles/blob/main/SPLIT-CLAUDE-MD-PLAN.md)
