"""Every tool that saves a modality checks its target before it stores it.

A save over a file the session loaded is refused (RecordedInputError). A tool
must not reach that refusal after it has already replaced a modality in memory:
before ``store_modality`` it either takes a result name no loaded file has
(``unloaded_name``) or checks a name it was given (``assert_save_target``). New
tools are held to it here; a site that needs neither is listed with its reason.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CHECKS = ("unloaded_name(", "assert_save_target(", "_unloaded_result_name(")
# (file relative to the checkout, function) -> why it needs no check
ALLOWED = {}


def _save_sites():
    for path in sorted((ROOT / "packages").glob("*/lobster/**/*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "save_modality"
            ):
                continue
            enclosing = min(
                (
                    f
                    for f in functions
                    if f.lineno <= node.lineno <= (f.end_lineno or f.lineno)
                ),
                key=lambda f: (f.end_lineno or f.lineno) - f.lineno,
            )
            before = "\n".join(
                source.splitlines()[enclosing.lineno - 1 : node.lineno - 1]
            )
            yield path.relative_to(ROOT).as_posix(), enclosing.name, node.lineno, before


SITES = list(_save_sites())


def test_the_audit_finds_the_save_sites():
    assert len(SITES) >= 30


@pytest.mark.parametrize(
    "file, function, line, before",
    SITES,
    ids=[f"{file.split('/')[-1]}:{line}" for file, _, line, _ in SITES],
)
def test_a_save_site_checks_its_target_first(file, function, line, before):
    if (file, function) in ALLOWED:
        return
    assert any(check in before for check in CHECKS), (
        f"{file}:{line} ({function}) saves a modality with no unloaded_name() or "
        "assert_save_target() before it; add one, or list it in ALLOWED with why"
    )
