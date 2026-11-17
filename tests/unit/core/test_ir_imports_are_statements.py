"""Every ``imports=`` entry of an IR is an import statement.

The exporter writes the entries as lines of the notebook's imports cell, so a bare
module name there is a ``NameError`` that stops the notebook.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]


def _literal_entries(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "imports":
            value = node.value
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "imports" for t in node.targets
        ):
            value = node.value
        else:
            continue
        if isinstance(value, (ast.List, ast.Tuple)):
            for item in value.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    yield item


def test_no_ir_lists_a_bare_name_as_an_import():
    bare = []
    for top in ("lobster", "packages"):
        for path in (ROOT / top).rglob("*.py"):
            if "tests" in path.parts or ".venv" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for item in _literal_entries(tree):
                text = item.value.strip()
                if text and not text.startswith(("import ", "from ", "#")):
                    bare.append(f"{path.relative_to(ROOT)}:{item.lineno}: {text}")
    assert bare == []
