---
phase: 05-python-integration
plan: 01
subsystem: interaction
tags: [llm, pydantic, structured-output, component-mapper, hitl]

requires:
  - phase: 02-charm-v2-migration
    provides: HITL component schemas and mapper infrastructure
provides:
  - Hybrid LLM/rule-based component selection in map_question()
  - ask_user factory pattern with LLM closure
  - ComponentSelection validator for invalid components and empty fallback
affects: [05-python-integration]

tech-stack:
  added: []
  patterns: [factory-with-closure, hybrid-rule-llm-selection, pydantic-model-validator]

key-files:
  created: []
  modified:
    - lobster/services/interaction/component_mapper.py
    - lobster/services/interaction/component_schemas.py
    - lobster/tools/user_interaction.py
    - lobster/agents/graph.py
    - tests/unit/services/test_component_mapper.py
    - tests/unit/tools/test_user_interaction.py

key-decisions:
  - "LLM path inserted after confirm check, before text_input fallback -- preserves all rule-based fast paths"
  - "Factory pattern with module-level backward-compat instance for ask_user"
  - "Pydantic model_validator corrects invalid component names to text_input automatically"

patterns-established:
  - "Hybrid selection: rule-based fast path for structural context, LLM for ambiguous"
  - "Tool factory with LLM closure: create_X_tool(llm=) pattern"

requirements-completed: [PYTH-01]

duration: 4min
completed: 2026-03-07
---

# Phase 5 Plan 1: LLM-Driven Component Selection Summary

**Hybrid LLM/rule-based component mapper with ask_user factory accepting supervisor LLM via closure**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-07T07:52:17Z
- **Completed:** 2026-03-07T07:56:37Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Hybrid component selection: rule-based fast path (options/clusters/threshold/confirm) unchanged, LLM structured output for ambiguous questions
- ComponentSelection Pydantic validator auto-corrects invalid component names and empty fallback_prompt
- ask_user converted to factory pattern with LLM closure, wired to supervisor_model in graph.py
- 28 total tests passing (21 mapper + 7 user_interaction), full backward compatibility

## Task Commits

Each task was committed atomically (TDD: RED then GREEN):

1. **Task 1: LLM path to component_mapper + ComponentSelection validator**
   - `ac31d0f` (test: failing tests for LLM-driven component selection)
   - `ecb5f21` (feat: hybrid LLM/rule-based component selection)
2. **Task 2: Convert ask_user to factory + wire LLM in graph.py**
   - `bf8df25` (test: failing tests for ask_user factory pattern)
   - `9f8dd46` (feat: convert ask_user to factory + wire supervisor LLM)

## Files Created/Modified

- `lobster/services/interaction/component_schemas.py` - Added model_validator for component name and fallback_prompt
- `lobster/services/interaction/component_mapper.py` - Added optional llm param, _llm_select_component with structured output
- `lobster/tools/user_interaction.py` - Replaced standalone tool with create_ask_user_tool(llm) factory
- `lobster/agents/graph.py` - Wired supervisor_model into _build_supervisor_tools -> ask_user
- `tests/unit/services/test_component_mapper.py` - 11 new tests (fast path, LLM path, validators, errors)
- `tests/unit/tools/test_user_interaction.py` - 4 new tests (factory, metadata, LLM pass-through, backward compat)

## Decisions Made

- LLM branch inserted after confirm pattern check, before text_input fallback -- all 5 existing rule-based paths preserved unchanged
- Factory pattern with module-level `ask_user = create_ask_user_tool()` for backward compatibility (existing imports unaffected)
- Pydantic model_validator (mode="after") corrects invalid component names to text_input and defaults empty fallback_prompt from data["question"]

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Component mapper now supports LLM-driven selection for novel question patterns
- ask_user tool receives supervisor LLM via closure, enabling richer interactive questions
- Ready for additional Python integration work in subsequent phase 5 plans

---
*Phase: 05-python-integration*
*Completed: 2026-03-07*
