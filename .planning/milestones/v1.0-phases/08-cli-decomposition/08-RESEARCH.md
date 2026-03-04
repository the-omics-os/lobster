# Phase 8: CLI Decomposition - Research

**Researched:** 2026-03-04
**Domain:** Python CLI monolith decomposition / Typer framework / module extraction
**Confidence:** HIGH

## Summary

Phase 8 decomposes `cli.py` (9,323 LOC, 77 top-level functions/classes) into focused modules under `cli_internal/commands/` while preserving the exact CLI behavior. The file currently serves as both the Typer app wiring layer AND the implementation of every command body. A significant portion of the work has already been done -- `cli_internal/` already contains 7,935 lines across 15 modules (light/ and heavy/ subpackages) covering queue, metadata, workspace, pipeline, config, file, agent, scaffold, validate, vector-search, purge, data, modality, and visualization commands.

What remains in `cli.py` are the "big four" commands (`init` at 1,355 LOC, `chat` at 368 LOC, `query` at 263 LOC, `_execute_command` slash-command dispatcher at 968 LOC), plus supporting infrastructure (client initialization, animations, shell execution, agent selection, streaming, session display). These are the heaviest and most coupled functions -- they share module-level state (`console`, `console_manager`, `app`, `config_app`) and cross-call each other extensively.

**Primary recommendation:** Extract remaining command bodies into 4-5 new modules under `cli_internal/commands/heavy/` (for init, chat, query) and `cli_internal/commands/` (for slash-command dispatch and shared session infrastructure). Pass `console`/`console_manager` as parameters or via a shared context object. Leave `cli.py` as pure Typer wiring: app definition, `@app.command()` decorators with parameter declarations, and one-line delegation calls.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLID-01 | Command bodies moved to cli_internal/commands/ | 77 functions catalogued; 22 Typer commands identified; extraction targets mapped to modules below |
| CLID-02 | cli.py reduced to composition/wiring with minimal control flow | Architecture pattern defines "thin wrapper" contract -- each `@app.command()` function delegates to a single function call |
| CLID-03 | All CLI subcommands work identically after decomposition | Verification via `lobster --help`, `lobster chat --help`, `lobster query --help`, `lobster init --help` output comparison + existing test suite |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.12+ | Language runtime | Project requirement |
| typer | installed | CLI framework | Already in use, all commands use `@app.command()` |
| rich | installed | Terminal output | Already in use, `Console`, `Panel`, `Table`, etc. |
| pytest | 9.0.1 | Test framework | Already in use |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| prompt_toolkit | installed (optional) | Interactive input with autocomplete | Used by `chat` command for REPL input |
| unittest.mock | stdlib | Test mocking | Patch paths change when code moves |

No new dependencies needed. This is pure internal restructuring.

## Architecture Patterns

### Current cli.py Function-to-Module Mapping

The 77 functions in `cli.py` cluster into clear domains. Functions already extracted to `cli_internal/` are excluded.

#### Module 1: `cli_internal/commands/heavy/init_commands.py` (~1,800 LOC)
The `init` command and all its helpers:
- `init()` -- main entry, 1,355 LOC (line 4898)
- `_create_workspace_config()` -- workspace config writer (line 4378, 56 LOC)
- `_create_global_config()` -- global config writer (line 4434, 76 LOC)
- `_uv_tool_env_handoff()` -- uv tool environment detection (line 4510, 104 LOC)
- `_ensure_provider_installed()` -- provider package check (line 4614, 35 LOC)
- `_prompt_docling_install()` -- docling optional install (line 4649, 42 LOC)
- `_download_ontology_databases()` -- ontology DB setup (line 4691, 46 LOC)
- `_prompt_smart_standardization()` -- smart standardization prompt (line 4737, 114 LOC)
- `_install_extended_data()` -- extended data setup (line 4851, 29 LOC)
- `_ensure_tui_installed()` -- TUI dependency check (line 4880, 18 LOC)
- `_get_installed_agents()` -- installed agent discovery (line 1269, 7 LOC)
- `_display_agent_selection_list()` -- agent list display (line 1276, 31 LOC)
- `_prompt_manual_agent_selection()` -- manual agent picker (line 1307, 107 LOC)
- `_prompt_automatic_agent_selection()` -- auto agent picker (line 1414, 63 LOC)
- `_check_and_prompt_install_packages()` -- package install prompt (line 1477, 79 LOC)
- `_save_agent_config()` -- agent config persistence (line 1556, 13 LOC)
- `_perform_agent_selection_non_interactive()` -- CI/CD agent selection (line 1569, 83 LOC)
- `_perform_agent_selection_interactive()` -- interactive agent selection (line 1652, 24 LOC)

**Dependencies:** console, console_manager, component_registry, provider_setup, LobsterAgentConfigurator

#### Module 2: `cli_internal/commands/heavy/chat_commands.py` (~1,200 LOC)
The `chat` REPL and its supporting infrastructure:
- `chat()` -- main entry, 368 LOC (line 6332)
- `init_client()` -- client initialization, 343 LOC (line 1832)
- `init_client_with_animation()` -- animated client init (line 3654, 31 LOC)
- `get_user_input_with_editing()` -- prompt_toolkit input (line 2175, 103 LOC)
- `_display_streaming_response()` -- streaming output handler (line 6253, 79 LOC)
- `display_session()` -- session history display (line 3523, 27 LOC)
- `display_welcome()` -- welcome banner (line 2825, 10 LOC)
- `display_goodbye()` -- exit banner (line 3126, 9 LOC)
- `_show_workspace_prompt()` -- workspace selection prompt (line 3550, 104 LOC)
- `change_mode()` -- agent/mode switching (line 1188, 81 LOC)
- `get_current_agent_name()` -- current agent display name (line 2666, 25 LOC)
- `handle_command()` -- slash command handler (line 6700, 55 LOC)

**Dependencies:** console, console_manager, LobsterClientAdapter, CommandClient, animations, _execute_command

#### Module 3: `cli_internal/commands/heavy/query_commands.py` (~300 LOC)
The `query` single-turn command:
- `query()` -- main entry, 263 LOC (line 7784)

**Dependencies:** console, console_manager, init_client, _display_streaming_response

#### Module 4: `cli_internal/commands/heavy/slash_commands.py` (~1,300 LOC)
Slash command dispatch (the `/help`, `/data`, etc. handler during chat):
- `_execute_command()` -- main dispatcher, 968 LOC (line 6780)
- `_dispatch_command()` -- shared dispatch table, 288 LOC (line 8141)
- `_extract_argument()` -- argument parser (line 6755, 23 LOC)
- `_command_files()` -- /files handler (line 8047, 45 LOC)
- `_command_save()` -- /save handler (line 8092, 21 LOC)
- `_command_restore()` -- /restore handler (line 8113, 28 LOC)
- `extract_available_commands()` -- command discovery, 537 LOC (line 651)
- `check_for_missing_slash_command()` -- typo detector (line 632, 19 LOC)
- `show_default_help()` -- /help output (line 3135, 60 LOC)

**Dependencies:** console, console_manager, client, all cli_internal command functions

#### Module 5: `cli_internal/commands/heavy/session_infra.py` (~800 LOC)
Shared session infrastructure used by chat, query, and commands:
- `LobsterClientAdapter` -- client wrapper class (line 317, 64 LOC)
- `CloudAwareCache` -- cache management class (line 381, 42 LOC)
- `CommandClient` -- command client class (line 568, 64 LOC)
- `NoOpProgress` -- progress stub class (line 218, 22 LOC)
- `should_show_progress()` -- progress gate (line 240, 41 LOC)
- `create_progress()` -- progress factory (line 281, 36 LOC)
- `_add_command_to_history()` -- history logging (line 423, 87 LOC)
- `_backup_command_to_file()` -- history file backup (line 510, 58 LOC)
- `_str_to_bool()` -- utility (line 1780, 11 LOC)
- `_resolve_profile_timings_flag()` -- utility (line 1791, 7 LOC)
- `_collect_profile_timings()` -- timing collection (line 1798, 12 LOC)
- `_maybe_print_timings()` -- timing display (line 1810, 22 LOC)
- `_get_extraction_cache_manager()` -- cache lazy loader (line 159, 7 LOC)

#### Module 6: `cli_internal/commands/heavy/animations.py` (~500 LOC)
Terminal animations (DNA helix, agent loading, exit):
- `_dna_helix_animation()` -- helix animation, 134 LOC (line 2691)
- `_dna_agent_loading_phase()` -- agent loading animation, 151 LOC (line 2835)
- `_dna_exit_animation()` -- exit animation, 140 LOC (line 2986)

#### Module 7: `cli_internal/commands/heavy/display_helpers.py` (~300 LOC)
Data display and formatting:
- `_format_data_preview()` -- matrix preview (line 3195, 47 LOC)
- `_format_dataframe_preview()` -- DataFrame preview (line 3242, 50 LOC)
- `_format_array_info()` -- array info (line 3292, 19 LOC)
- `_get_matrix_info()` -- matrix info extraction (line 3311, 25 LOC)
- `_display_status_info()` -- status display, 187 LOC (line 3336)

#### Remaining in cli.py: Config subcommands (~600 LOC)
These `@config_app.command()` functions should move to `cli_internal/commands/light/config_commands.py` (which already exists at 754 LOC):
- `config_callback()` -- config app callback (line 1694)
- `config_test()` / `test()` -- config test, 304 LOC (line 3685/8993)
- `list_models()` -- model listing (line 8668)
- `list_profiles()` -- profile listing (line 8707)
- `config_show_subcommand()` -- show subcommand (line 8722)
- `show_config()` -- show-config subcommand, 234 LOC (line 8759)
- `create_custom()` -- create custom config (line 9175, 71 LOC)
- `generate_env()` -- env generation (line 9246, 78 LOC)

#### Remaining Typer commands:
- `status()` -- trivial wrapper (line 3989, 6 LOC)
- `metadata_command()` -- delegating wrapper (line 3995, 83 LOC)
- `activate()` -- workspace activation (line 4078, 127 LOC)
- `deactivate()` -- workspace deactivation (line 4205, 83 LOC)
- `purge()` -- purge command (line 4288, 90 LOC)
- `execute_shell_command()` -- shell exec for /shell, 388 LOC (line 2278)
- `command_cmd()` -- lobster command subcommand, 154 LOC (line 8429)
- `vector_search_cmd()` -- vector search, 42 LOC (line 8583)
- `serve()` -- serve command, 43 LOC (line 8625)
- `dashboard_command()` -- dashboard, 36 LOC (line 7748)
- `default_callback()` -- app callback (line 1741)

### Target cli.py Structure After Decomposition

```python
# cli.py (~300-400 LOC) -- WIRING ONLY
"""Typer app definition and command registration."""

import typer
from lobster.version import __version__

# App definition
app = typer.Typer(...)
config_app = typer.Typer(...)
app.add_typer(config_app, name="config")
app.add_typer(agents_app, name="agents")
app.add_typer(scaffold_app, name="scaffold")
app.add_typer(validate_app, name="validate-plugin")

# Each command: decorator + parameter declarations + single delegation call
@app.command()
def chat(workspace: ..., session_id: ..., ...):
    """Start interactive chat session."""
    from lobster.cli_internal.commands.heavy.chat_commands import chat_impl
    chat_impl(workspace=workspace, session_id=session_id, ...)

@app.command()
def init(global_config: ..., force: ..., ...):
    """Initialize Lobster AI configuration."""
    from lobster.cli_internal.commands.heavy.init_commands import init_impl
    init_impl(global_config=global_config, force=force, ...)
```

### Shared State Pattern

The major coupling issue is that many functions reference module-level `console` and `console_manager`. The existing pattern (already used in `cli_internal/commands/light/`) is:

1. **Each module creates its own `console`** via `Console()` or `get_console_manager()`
2. **Functions that need client access** receive it as a parameter
3. **The OutputAdapter pattern** abstracts console output for reusability

For the heavy commands that need both console and client:
```python
# In cli_internal/commands/heavy/chat_commands.py
from lobster.ui.console_manager import get_console_manager

console_manager = get_console_manager()
console = console_manager.console

def chat_impl(workspace, session_id, reasoning, verbose, debug, ...):
    """Full chat implementation."""
    ...
```

This matches the existing pattern in `agent_commands.py` (line 47: `console = Console()`).

### Anti-Patterns to Avoid

- **Passing `app` or `config_app` to extracted modules:** Typer decorators must stay in `cli.py`. Only the function body moves.
- **Circular imports:** `cli.py` imports from `cli_internal/`, never the reverse. If extracted code needs something from `cli.py`, it should be in `cli_internal/` too.
- **Breaking `lobster.cli:app` entry point:** `pyproject.toml` declares `lobster = "lobster.cli:app"`. The `app` object must remain in `cli.py`.
- **Moving Typer parameter declarations:** The `typer.Option(...)` annotations define the CLI interface and must stay with the `@app.command()` decorator in `cli.py`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Console output abstraction | Custom output protocol | Existing `OutputAdapter` / `ConsoleOutputAdapter` | Already in `cli_internal/commands/output_adapter.py`, used by all light/heavy commands |
| Module-level console sharing | Singleton pattern or global state passing | `get_console_manager()` from `lobster.ui.console_manager` | Already used by 12+ existing modules, provides themed console |
| Lazy imports | Custom import mechanism | Existing `__getattr__` pattern in `cli_internal/commands/__init__.py` | Already established for heavy commands |

## Common Pitfalls

### Pitfall 1: Circular Import Between cli.py and Extracted Modules
**What goes wrong:** Moving `init_client()` to a module that imports something still in `cli.py` creates a cycle.
**Why it happens:** Functions in cli.py call each other extensively (e.g., `chat` calls `init_client` calls `display_welcome`).
**How to avoid:** Extract ALL functions that a command body depends on into the same module or a shared infrastructure module. The dependency graph is: `chat_commands` depends on `session_infra` + `slash_commands` + `animations`. Never have an extracted module import from `lobster.cli`.
**Warning signs:** `ImportError` at startup.

### Pitfall 2: Test Patch Paths Break
**What goes wrong:** Existing tests patch `lobster.cli._add_command_to_history` -- after move, patch target changes.
**Why it happens:** `mock.patch` uses string paths that must match the actual import location.
**How to avoid:** Update ALL test patch paths. Search for `"lobster.cli."` in test files. The 3 CLI test files (1,003 LOC total) need patch path updates.
**Warning signs:** Tests pass but mock doesn't activate (assertions on mock fail).

### Pitfall 3: Module-Level Side Effects on Import
**What goes wrong:** `cli.py` has significant module-level setup (pandas config, warning filters, multiprocessing fork mode, logging levels). Moving code to new modules risks duplicating or losing these.
**How to avoid:** Keep ALL module-level configuration in `cli.py`. Extracted modules should be pure implementation -- no side effects on import.
**Warning signs:** Different behavior when running `lobster chat` vs importing the module directly.

### Pitfall 4: Typer Parameter Signature Mismatch
**What goes wrong:** The `@app.command()` decorator inspects function signatures for CLI argument generation. If the wrapper in `cli.py` has different parameter names or types than the impl function, Typer generates wrong CLI.
**How to avoid:** Keep the full Typer parameter declarations in `cli.py`. The impl function can use `**kwargs` or match the exact signature. The cleanest pattern is matching parameter names exactly.
**Warning signs:** `lobster chat --help` shows different options.

### Pitfall 5: `_execute_command` Couples to Everything
**What goes wrong:** The 968-line `_execute_command()` function directly calls 40+ functions from both `cli.py` and `cli_internal/`. Extracting it requires resolving all these dependencies.
**Why it happens:** It's the central slash-command dispatcher during chat.
**How to avoid:** Extract it along with `_dispatch_command()` (which already delegates to `cli_internal/` functions) into a single `slash_commands.py` module. Functions it calls that are still in `cli.py` must also be extracted first.

## Code Examples

### Pattern: Thin Wrapper in cli.py

```python
# cli.py -- after decomposition
@app.command()
def chat(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", ...),
    session_id: Optional[str] = typer.Option(None, "--session-id", "-s", ...),
    reasoning: bool = typer.Option(False, "--reasoning", ...),
    # ... all other params with typer.Option declarations ...
):
    """Start an interactive chat session with the multi-agent system."""
    from lobster.cli_internal.commands.heavy.chat_commands import chat_impl
    chat_impl(
        workspace=workspace,
        session_id=session_id,
        reasoning=reasoning,
        # ... pass all params through ...
    )
```

### Pattern: Extracted Implementation Module

```python
# cli_internal/commands/heavy/chat_commands.py
"""Chat command implementation -- interactive REPL session."""

from pathlib import Path
from typing import Optional

from lobster.ui.console_manager import get_console_manager

console_manager = get_console_manager()
console = console_manager.console


def chat_impl(
    workspace: Optional[Path],
    session_id: Optional[str],
    reasoning: bool,
    verbose: bool,
    debug: bool,
    # ... all params ...
):
    """Full chat implementation body (moved from cli.py)."""
    from lobster.cli_internal.commands.heavy.session_infra import (
        init_client_with_animation,
        LobsterClientAdapter,
    )
    # ... implementation ...
```

### Pattern: Test Patch Path Update

```python
# Before (test patches cli.py directly):
with patch("lobster.cli._add_command_to_history", return_value=True):
    ...

# After (patches new location):
with patch("lobster.cli_internal.commands.heavy.session_infra._add_command_to_history", return_value=True):
    ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Monolith cli.py (9,323 LOC) | Partial extraction (7,935 LOC in cli_internal/) | Pre-Phase 8 | ~46% of logic already extracted |
| All commands inline | light/heavy split with lazy loading | Pre-Phase 8 | Startup time improved ~5s |

**Already extracted (no work needed):**
- Queue commands (945 LOC)
- Metadata commands (1,077 LOC)
- Workspace commands (930 LOC)
- Config commands (754 LOC)
- Pipeline commands (371 LOC)
- File commands (705 LOC)
- Purge commands (468 LOC)
- Agent commands (466 LOC)
- Data commands (224 LOC)
- Modality commands (578 LOC)
- Visualization commands (351 LOC)
- Scaffold commands (75 LOC)
- Validate commands (59 LOC)
- Vector search commands (75 LOC)
- Output adapter (328 LOC)
- Path resolution (291 LOC)

**Still in cli.py (work needed):**
- `init` command body + 17 helpers (~1,800 LOC)
- `chat` command body + session infrastructure (~1,200 LOC)
- `query` command body (~300 LOC)
- Slash command dispatch (~1,300 LOC)
- Animations (~500 LOC)
- Display helpers (~300 LOC)
- Config subcommands (~600 LOC)
- Shell execution (~400 LOC)
- Remaining small commands (activate, deactivate, status, metadata, serve, dashboard, command) (~700 LOC)

**Total to extract:** ~7,100 LOC
**Target cli.py after:** ~300-400 LOC (Typer wiring only)

## Open Questions

1. **Should `execute_shell_command` (388 LOC) go into slash_commands or session_infra?**
   - What we know: It's called from `_execute_command()` for the `/shell` slash command
   - Recommendation: Put it in `slash_commands.py` since it's only invoked from slash command dispatch

2. **Should config subcommands stay as separate `@config_app.command()` in cli.py or move entirely?**
   - What we know: Some config logic is already in `cli_internal/commands/light/config_commands.py`. The `@config_app.command()` functions in cli.py mostly do wiring + Typer params + delegation.
   - Recommendation: Move bodies to config_commands.py; keep `@config_app.command()` decorators in cli.py as thin wrappers (consistent with all other commands)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.1 |
| Config file | `pytest.ini` |
| Quick run command | `cd /Users/tyo/Omics-OS/lobster && python -m pytest tests/unit/cli/ -x --no-cov -q` |
| Full suite command | `cd /Users/tyo/Omics-OS/lobster && python -m pytest tests/unit/cli/ tests/integration/test_session_provenance.py --no-cov -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLID-01 | Command bodies live in cli_internal/commands/ | unit | `python -m pytest tests/unit/cli/test_cli_decomposition.py -x --no-cov -q` | No -- Wave 0 |
| CLID-02 | cli.py contains only wiring (LOC check, no business logic) | unit | `python -m pytest tests/unit/cli/test_cli_decomposition.py::test_cli_is_wiring_only -x --no-cov -q` | No -- Wave 0 |
| CLID-03 | All CLI subcommands work identically | smoke | `lobster --help && lobster chat --help && lobster query --help && lobster init --help` | Manual |

### Sampling Rate
- **Per task commit:** `cd /Users/tyo/Omics-OS/lobster && python -m pytest tests/unit/cli/ -x --no-cov -q`
- **Per wave merge:** `cd /Users/tyo/Omics-OS/lobster && python -m pytest tests/unit/cli/ tests/integration/test_session_provenance.py --no-cov -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/cli/test_cli_decomposition.py` -- structural tests: cli.py LOC < 500, no function bodies > 10 lines (excluding Typer param declarations), all heavy modules importable
- [ ] Existing test patch paths in `tests/unit/cli/test_bug_fixes_security_logging.py` must be updated for new module locations

## Sources

### Primary (HIGH confidence)
- Direct analysis of `/Users/tyo/Omics-OS/lobster/lobster/cli.py` (9,323 LOC) -- line-by-line function inventory
- Direct analysis of `/Users/tyo/Omics-OS/lobster/lobster/cli_internal/` (7,935 LOC across 15 modules) -- existing extraction pattern
- Phase 4 research precedent (`.planning/phases/04-geo-service-decomposition/04-RESEARCH.md`) -- decomposition methodology
- Existing test files in `tests/unit/cli/` (1,003 LOC) -- patch path dependencies

### Secondary (MEDIUM confidence)
- Typer documentation for decorator/callback behavior -- based on training data (Typer is stable, API unchanged)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, pure restructuring
- Architecture: HIGH -- existing cli_internal pattern is well-established with 15 modules as precedent
- Pitfalls: HIGH -- based on direct code analysis and Phase 4 decomposition experience

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (stable domain, no external dependencies)
