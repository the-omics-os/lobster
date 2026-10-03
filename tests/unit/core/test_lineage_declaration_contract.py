"""Contract: tools that derive a modality must declare its processing step.

The test uses AST inspection to ensure calls provide ``step=`` explicitly. This prevents
new suffixes from silently relying on inferred lineage.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Modules whose `store_modality` calls must pass an explicit `step=`.
#: Extend this list as tools are audited — see the module docstring.
DECLARING_MODULES = [
    "packages/lobster-transcriptomics/lobster/agents/transcriptomics/shared_tools.py",
]


def _store_modality_calls(path: Path) -> list[ast.Call]:
    """Every `store_modality(...)` call in a module, via AST rather than grep.

    AST because a regex over source cannot tell a real call from the same text inside a
    docstring or an error message, and this file's whole job is to be trustworthy.
    """
    tree = ast.parse(path.read_text())
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name == "store_modality":
            calls.append(node)
    return calls


@pytest.mark.contract
@pytest.mark.parametrize("rel_path", DECLARING_MODULES)
def test_store_modality_calls_declare_their_step(rel_path):
    """A derived modality must declare `step=`, not leave it to name inference."""
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"missing module: {rel_path}"

    calls = _store_modality_calls(path)
    assert calls, f"no store_modality calls found in {rel_path} — has it moved?"

    undeclared = [
        call.lineno
        for call in calls
        if "step" not in {kw.arg for kw in call.keywords if kw.arg}
    ]
    assert not undeclared, (
        f"{rel_path}: store_modality without an explicit step= at line(s) "
        f"{undeclared}. Name-suffix inference reads only the TERMINAL suffix, so a new "
        f"or unregistered suffix silently records the wrong processing_step. "
        f"Pass step=<one of CANONICAL_STEPS>."
    )


@pytest.mark.contract
@pytest.mark.parametrize("rel_path", DECLARING_MODULES)
def test_declared_steps_are_canonical(rel_path):
    """A declared step outside CANONICAL_STEPS widens the vocabulary silently.

    `CANONICAL_STEPS` is documented as "guidance, not enforcement — any string is a valid
    step", which is fine for agent packages defining domain steps. But a *typo* is also a
    valid string, and would read as an unknown stage to any consumer. For the modules in
    this contract, the step must be one of the canonical set.
    """
    from lobster.core.provenance.lineage import CANONICAL_STEPS

    path = REPO_ROOT / rel_path
    bad = []
    for call in _store_modality_calls(path):
        for kw in call.keywords:
            if kw.arg == "step" and isinstance(kw.value, ast.Constant):
                if kw.value.value not in CANONICAL_STEPS:
                    bad.append((call.lineno, kw.value.value))
    assert not bad, f"{rel_path}: non-canonical step(s) {bad}"


@pytest.mark.contract
def test_declared_step_wins_over_name_inference():
    """Explicit declarations take precedence over suffix-based inference.

    If inference overrode the declared step, a call site could still pass the static contract
    while recording the wrong stage.
    """
    from lobster.core.provenance.lineage import create_lineage_metadata

    lineage = create_lineage_metadata(
        modality_name="pbmc3k_something_unregistered",
        parent_modality="pbmc3k",
        step="reduced",
    )
    assert lineage.processing_step == "reduced"
