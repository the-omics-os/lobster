# How to Split a Large CLAUDE.md

Tutorial for agents. Follow these steps to reduce a bloated CLAUDE.md to ~150-300 lines.

## Why Split

Claude Code loads CLAUDE.md every session. Large files (>300 lines) cause:
- Instruction "forgetting" as context fills
- Reduced adherence to rules buried in long text
- Wasted tokens on rarely-needed details

Target: Main file <300 lines. Details in @imported docs loaded on-demand.

## Step 1: Analyze Current Structure

Count lines and identify sections:
```bash
wc -l CLAUDE.md
grep -n "^## " CLAUDE.md
```

Categorize each section:
- KEEP INLINE: Rules, commands, personality (needed every session)
- EXTRACT: Architecture, patterns, detailed reference (needed sometimes)

## Step 2: Create Directory Structure

```bash
mkdir -p .claude/docs
```

## Step 3: Extract Detailed Content

For each section to extract, create a focused doc in .claude/docs/:

```
.claude/docs/
├── architecture.md      # Data flow, diagrams, component interactions
├── code-layout.md       # File structure, where things live
├── development-rules.md # Patterns, conventions, how to implement
└── tooling.md           # Setup, testing, CI/CD, publishing
```

Each extracted doc should:
- Have a clear single purpose
- Be self-contained (readable without main CLAUDE.md)
- Include only content relevant to its topic

## Step 4: Rewrite Main CLAUDE.md

Structure for the lean main file:

```markdown
# CLAUDE.md

[1-2 sentence project description]

## Identity/Personality (if applicable)
[Keep inline - defines assistant behavior]

## Project Overview
[~20 lines max - what the project does, core concepts]

## Hard Rules
[Keep inline - non-negotiable rules that must always be visible]

## Quick Commands
[Most frequently used commands only]

## Key Architecture (Summary)
[Critical flow in 1 line, key files list, brief component table]
For details: @.claude/docs/architecture.md

## Essential Patterns (if any)
[Only patterns needed in 90% of tasks]
For all patterns: @.claude/docs/development-rules.md

## Documentation Navigation
[Table mapping: Document → Load When → Examples]

**Decision rule**: If unsure, load the doc.
```

## Step 5: Add Import References

Use @path/to/file.md syntax to reference detailed docs:

```markdown
For detailed architecture: @.claude/docs/architecture.md
For complete file structure: @.claude/docs/code-layout.md
```

Claude Code loads these on-demand when referenced or when working in related areas.

## Step 6: Create Trigger Conditions

Add a table telling the agent when to load each doc:

```markdown
| Document | Load When | Examples |
|----------|-----------|----------|
| @.claude/docs/architecture.md | Modifying data flow | "Change how X connects to Y" |
| @.claude/docs/code-layout.md | Finding files | "Where is X?" |
```

End with: "If unsure, load the doc."

## Step 7: Verify

```bash
wc -l CLAUDE.md  # Should be <300
ls -la .claude/docs/  # Should have extracted docs
```

Keep original as backup:
```bash
cp CLAUDE.md CLAUDE.md.backup
```

## Checklist

Before:
- [ ] Backup original CLAUDE.md

Extract to .claude/docs/:
- [ ] Architecture/data flow content
- [ ] File structure/component reference
- [ ] Development patterns/conventions
- [ ] Tooling/environment setup

Keep inline:
- [ ] Project overview (brief)
- [ ] Hard rules (non-negotiable)
- [ ] Quick commands (most used)
- [ ] Personality/identity
- [ ] Documentation navigation with trigger table

After:
- [ ] Main file <300 lines
- [ ] Each extracted doc is self-contained
- [ ] @import references point to correct paths
- [ ] Trigger conditions table is clear

## Common Mistakes

1. Extracting too little - Be aggressive. If it's not needed every session, extract it.
2. Vague trigger conditions - Use concrete task phrases, not abstract descriptions.
3. Missing decision rule - Always add "If unsure, load the doc."
4. Duplicating content - Extract once, reference everywhere.
5. No backup - Always keep CLAUDE.md.backup until verified working.
