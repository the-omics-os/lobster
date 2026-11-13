"""List user-visible English literals that bypass ``h()`` / ``t()``.

Run on files to see what is left to translate::

    python tests/unit/i18n/literal_scan.py lobster/cli_internal/commands/light/queue_commands.py

The heuristic flags string literals (and f-string text) passed to output calls
such as ``console.print``, ``output.print``, Rich ``Table``/``Panel`` titles and
columns, block builders, ``typer.Option(help=...)`` and prompts. Logger calls,
``style=`` values and literals already inside ``h()``/``t()`` are ignored.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

SINK_NAMES = {
    "print",
    "echo",
    "secho",
    "add_column",
    "add_row",
    "Panel",
    "Table",
    "Text",
    "Rule",
    "confirm",
    "ask",
    "prompt",
    "section_block",
    "alert_block",
    "hint_block",
    "list_block",
    "kv_block",
    "table_block",
    "_send_alert",
}
TEXT_KEYWORDS = {"help", "title", "caption", "message", "description", "subtitle"}
SKIP_KEYWORDS = {"style", "border_style", "header_style", "justify", "box", "end"}
WRAPPERS = {"h", "t"}
WORD = re.compile(r"[A-Za-z]{2,}")
KEEP = re.compile(
    r"^\s*("
    r"[/\-]{0,2}[a-z0-9_.\-]+"  # identifiers, flags, commands
    r"|https?://\S+"
    r"|\[/?[a-z ]+\]"  # bare Rich markup
    r"|[A-Z_]{2,}"  # env vars, constants
    r")\s*$"
)
MARKUP = re.compile(r"\[/?[a-z0-9 #_.:\-]*\]")


def _is_ui_text(text: str) -> bool:
    bare = MARKUP.sub("", text).strip()
    return bool(bare) and bool(WORD.search(bare)) and not KEEP.match(bare)


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _text_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) else "{}"
            for part in node.values
        )
    return None


class _Scanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.found: List[Tuple[int, str]] = []
        self._wrapped = 0

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        if name in WRAPPERS:
            self._wrapped += 1
            self.generic_visit(node)
            self._wrapped -= 1
            return
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id in {"logger", "logging", "log"}:
                return
        if not self._wrapped:
            candidates = []
            if name in SINK_NAMES:
                candidates.extend(node.args)
            for kw in node.keywords:
                if kw.arg in SKIP_KEYWORDS:
                    continue
                if kw.arg in TEXT_KEYWORDS or name in SINK_NAMES:
                    candidates.append(kw.value)
            for arg in candidates:
                text = _text_of(arg)
                if text is not None and _is_ui_text(text):
                    self.found.append((arg.lineno, text))
        self.generic_visit(node)


def scan(path: Path) -> List[Tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    scanner = _Scanner()
    scanner.visit(tree)
    return sorted(scanner.found)


def iter_calls(path: Path, names: set[str]) -> Iterator[Tuple[int, str, str]]:
    """Yield (line, function, literal) for ``h("...")``/``t("...")`` calls."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in names
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            yield node.lineno, node.func.id, node.args[0].value


def main(argv: List[str]) -> int:
    total = 0
    for name in argv:
        for line, text in scan(Path(name)):
            print(f"{name}:{line}: {text!r}")
            total += 1
    print(f"{total} untranslated literal(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
