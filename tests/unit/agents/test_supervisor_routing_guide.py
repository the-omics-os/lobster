"""Regression smoke set for supervisor task routing (TICKET_TRIAGE Phase 1D).

Context
-------
The supervisor was over-routing to ``execute_custom_code`` (90%+ across all
models) instead of delegating to specialized agents. Root causes (verified by
Codex #10 + research/D_alejandro_engine_bugs.md):

  * No task->agent routing guide in the supervisor prompt.
  * Category D framed ``execute_custom_code`` as a delegation peer, not a
    last resort.
  * The prompt directory rendered the short ``description`` field, never the
    richer ``handoff_tool_description`` that carries domain routing keywords.
  * ``visualization_expert_agent`` had a vacuous handoff description.

How this test proves the fix
----------------------------
Routing is ultimately an LLM decision, so we cannot assert model behavior
deterministically in a unit test. Instead we assert on the *prompt contract*
the LLM reads: we parse the ``<Task Routing Guide>`` section the fix adds and
run a simple, transparent keyword-overlap router over it. For a representative
task, the router must select the SPECIALIZED agent, not ``execute_custom_code``.

Before the fix the routing guide section does not exist, so the router falls
through to ``execute_custom_code`` for every task and the whole smoke set fails
(this is the required TDD red state). After the fix the guide carries each
specialist's ``handoff_tool_description`` and the router resolves each task to
its owning specialist.

This is a routing-correctness smoke set, NOT a benchmark. It makes no claim
about SCBench / LabBench2 numbers.
"""

from __future__ import annotations

from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

from lobster.agents.supervisor import create_supervisor_prompt
from lobster.config.agent_registry import get_agent_registry_config, get_worker_agents
from lobster.config.supervisor_config import SupervisorConfig

# Marker the fix must emit around the generated routing guide.
ROUTING_GUIDE_OPEN = "<Task Routing Guide>"
ROUTING_GUIDE_CLOSE = "</Task Routing Guide>"

# Sentinel returned when no specialist matches a task.
LAST_RESORT = "execute_custom_code"


# ---------------------------------------------------------------------------
# Representative smoke set: (label, natural-language task, salient domain
# keywords, expected specialist agent). One task per benchmark-relevant flow.
# ---------------------------------------------------------------------------
REPRESENTATIVE_TASKS: List[Tuple[str, str, List[str], str]] = [
    (
        "transcriptomics_qc",
        "Run quality control on my single-cell RNA-seq data: filter low-quality "
        "cells and genes, then normalize.",
        ["qc", "quality", "filtering", "normalization", "single-cell", "rna-seq"],
        "transcriptomics_expert",
    ),
    (
        "differential_expression",
        "Run pseudobulk differential expression between treated and control "
        "clusters in my single-cell RNA-seq data.",
        ["pseudobulk", "differential expression", "single-cell", "rna-seq"],
        "transcriptomics_expert",
    ),
    (
        "annotation",
        "Annotate the cell types for each cluster in my single-cell data.",
        ["cell type annotation", "cluster", "single-cell"],
        "transcriptomics_expert",
    ),
    (
        "literature_search",
        "Find published literature on CAR-T therapy and discover candidate "
        "datasets, extracting the analysis methods and parameters.",
        [
            "literature search",
            "dataset discovery",
            "method analysis",
            "parameter extraction",
        ],
        "research_agent",
    ),
]


def _build_prompt() -> str:
    """Render the supervisor prompt with data context disabled (hermetic)."""
    dm = MagicMock()
    dm.list_modalities.return_value = []
    dm.workspace_path = "/tmp/lobster-routing-smoke-nonexistent"

    config = SupervisorConfig()
    config.include_data_context = False
    config.include_system_info = False
    config.include_memory_stats = False

    active_agents = list(get_worker_agents().keys())
    return create_supervisor_prompt(
        dm, config, active_agents=active_agents, interactive=False
    )


def _extract_routing_guide(prompt: str) -> str:
    """Return the routing-guide block, or '' if the fix has not added it yet."""
    start = prompt.find(ROUTING_GUIDE_OPEN)
    if start == -1:
        return ""
    end = prompt.find(ROUTING_GUIDE_CLOSE, start)
    if end == -1:
        return ""
    return prompt[start : end + len(ROUTING_GUIDE_CLOSE)]


def _guide_entries(guide: str) -> dict[str, str]:
    """Parse the guide into {agent_name: lowercased entry text}.

    Entries are lines of the form ``- <agent_name>: <handoff_tool_description>``.
    """
    entries: dict[str, str] = {}
    known = set(get_worker_agents().keys())
    for raw in guide.splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        if ":" not in body:
            continue
        name, _, rest = body.partition(":")
        name = name.strip()
        if name in known:
            entries[name] = rest.strip().lower()
    return entries


def _simulate_routing(prompt: str, task_keywords: List[str]) -> str:
    """Deterministic proxy for the LLM's routing decision.

    Parse the routing guide, score every listed specialist by the number of
    distinct task keywords present in its entry, and return the unique
    argmax specialist. If the guide is absent or no specialist matches, return
    the last-resort sentinel ``execute_custom_code`` -- mirroring the buggy
    fall-through behavior this fix eliminates.
    """
    guide = _extract_routing_guide(prompt)
    if not guide:
        return LAST_RESORT

    entries = _guide_entries(guide)
    if not entries:
        return LAST_RESORT

    kws = [k.lower() for k in task_keywords]
    scores = {
        name: sum(1 for kw in kws if kw in text) for name, text in entries.items()
    }
    best = max(scores.values(), default=0)
    if best == 0:
        return LAST_RESORT

    winners = [name for name, score in scores.items() if score == best]
    if len(winners) != 1:
        # Ambiguous tie -> treat as unresolved (test will flag prompt weakness).
        return "AMBIGUOUS:" + ",".join(sorted(winners))
    return winners[0]


# ---------------------------------------------------------------------------
# The regression proof: each representative task routes to its specialist.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label,task,keywords,expected",
    REPRESENTATIVE_TASKS,
    ids=[t[0] for t in REPRESENTATIVE_TASKS],
)
def test_representative_task_routes_to_specialist(label, task, keywords, expected):
    if get_agent_registry_config(expected) is None:
        pytest.skip(f"{expected} not installed in this environment")

    prompt = _build_prompt()
    chosen = _simulate_routing(prompt, keywords)

    assert chosen != LAST_RESORT, (
        f"[{label}] supervisor prompt routes to execute_custom_code instead of "
        f"a specialist. Task: {task!r}"
    )
    assert chosen == expected, (
        f"[{label}] expected routing to {expected!r} but keyword-overlap over "
        f"the routing guide chose {chosen!r}. Task: {task!r}"
    )


def test_no_representative_task_falls_back_to_custom_code():
    """Aggregate guard: zero tasks in the smoke set hit the last resort."""
    prompt = _build_prompt()
    fell_back = [
        label
        for (label, _task, keywords, _expected) in REPRESENTATIVE_TASKS
        for chosen in [_simulate_routing(prompt, keywords)]
        if chosen == LAST_RESORT
    ]
    assert not fell_back, (
        f"{len(fell_back)}/{len(REPRESENTATIVE_TASKS)} representative tasks routed "
        f"to execute_custom_code: {fell_back}"
    )


# ---------------------------------------------------------------------------
# Structural guards on the prompt contract (secondary, but cheap regression net)
# ---------------------------------------------------------------------------
def test_routing_guide_section_present():
    prompt = _build_prompt()
    assert ROUTING_GUIDE_OPEN in prompt and ROUTING_GUIDE_CLOSE in prompt, (
        "supervisor prompt is missing the <Task Routing Guide> section"
    )


def test_execute_custom_code_framed_as_last_resort():
    prompt = _build_prompt().lower()
    assert "last resort" in prompt or "last-resort" in prompt, (
        "supervisor prompt no longer frames execute_custom_code as a last resort"
    )


def test_routing_guide_uses_rich_handoff_descriptions():
    """Guide must carry handoff_tool_description text, not just short descriptions."""
    prompt = _build_prompt()
    guide = _extract_routing_guide(prompt)
    assert guide, "routing guide section absent"

    tx = get_agent_registry_config("transcriptomics_expert")
    assert tx is not None and tx.handoff_tool_description
    # A distinctive phrase that lives ONLY in the handoff description, not the
    # short description -> proves the richer field is what feeds the guide.
    assert "not data_expert" in tx.handoff_tool_description.lower()
    assert "not data_expert" in guide.lower(), (
        "routing guide is not using handoff_tool_description "
        "(rich routing text missing)"
    )


def test_visualization_expert_handoff_description_enriched():
    """The vacuous visualization handoff description must gain domain vocabulary."""
    cfg = get_agent_registry_config("visualization_expert_agent")
    assert cfg is not None
    desc = (cfg.handoff_tool_description or "").lower()
    # Must no longer be the tautological one-liner.
    assert desc != "delegate visualization tasks to the visualization expert agent"
    # Must name concrete plot types the agent actually produces.
    assert sum(kw in desc for kw in ("umap", "heatmap", "violin", "dot plot")) >= 3, (
        "visualization handoff description lacks concrete plot-type vocabulary"
    )
