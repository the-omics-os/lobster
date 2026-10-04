"""Contract: wired tools must identify themselves in provenance.

Agent attribution is optional for backward compatibility, so this contract explicitly
checks the audited modules and requires each ``log_tool_usage`` call to pass ``agent=``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Modules whose `log_tool_usage` calls must pass an explicit `agent=`.
#: Extend as tools are audited — see the module docstring.
ATTRIBUTING_MODULES = [
    "packages/lobster-transcriptomics/lobster/agents/transcriptomics/shared_tools.py",
]


def _log_tool_usage_calls(path: Path) -> list[ast.Call]:
    """Every `log_tool_usage(...)` call, via AST rather than grep.

    AST because a regex cannot distinguish a real call from the same text inside a docstring
    or an error message, and this file's only job is to be trustworthy.
    """
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "attr", None) or getattr(node.func, "id", None))
        == "log_tool_usage"
    ]


@pytest.mark.contract
@pytest.mark.parametrize("rel_path", ATTRIBUTING_MODULES)
def test_wired_tools_pass_an_explicit_agent(rel_path):
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"missing module: {rel_path}"

    calls = _log_tool_usage_calls(path)
    assert calls, f"no log_tool_usage calls found in {rel_path} — has it moved?"

    missing = [
        call.lineno
        for call in calls
        if "agent" not in {kw.arg for kw in call.keywords if kw.arg}
    ]
    assert not missing, (
        f"{rel_path}: log_tool_usage without an explicit agent= at line(s) {missing}. "
        f"Without it the activity records agent='data_manager' — the object writing the "
        f"record, not the agent that ran the tool — provenance cannot identify its actor. "
    )


@pytest.mark.contract
@pytest.mark.parametrize("rel_path", ATTRIBUTING_MODULES)
def test_agent_comes_from_the_factory_parameter(rel_path):
    """It must be the threaded identity, not a hardcoded string.

    A literal would be attribution that cannot be wrong at review time but is wrong the moment
    the factory is reused by another agent — which is exactly how `create_shared_tools` is used
    (transcriptomics, proteomics, metabolomics and drug-discovery all call their own copy).
    """
    path = REPO_ROOT / rel_path
    hardcoded = []
    for call in _log_tool_usage_calls(path):
        for kw in call.keywords:
            if kw.arg == "agent" and isinstance(kw.value, ast.Constant):
                hardcoded.append((call.lineno, kw.value.value))
    assert not hardcoded, (
        f"{rel_path}: agent= is a literal at {hardcoded}. Pass the factory's agent_name "
        f"parameter so the identity follows the agent that owns the tools."
    )


@pytest.mark.contract
def test_default_is_still_the_historical_literal():
    """Backward compatibility is load-bearing: unwired call sites must not change behaviour."""
    from lobster.core.runtime.data_manager import DataManagerV2

    assert DataManagerV2.DEFAULT_PROVENANCE_AGENT == "data_manager"


@pytest.mark.contract
def test_agent_is_optional_on_log_tool_usage():
    """A required parameter would break ~116 call sites at once."""
    import inspect

    from lobster.core.runtime.data_manager import DataManagerV2

    param = inspect.signature(DataManagerV2.log_tool_usage).parameters["agent"]
    assert param.default is None, "agent must be optional; thread it incrementally"
