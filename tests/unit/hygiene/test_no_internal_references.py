"""Shipped code and tests carry no workshop references.

A comment keeps the reason for a choice, in neutral words. It does not name
who approved it, a planning folder, or a path on somebody's machine.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOTS = ("lobster", "packages", "tests", "lobster-tui", "skills")
SUFFIXES = {".py", ".go", ".md", ".json", ".toml", ".yaml", ".yml", ".txt", ".sh"}
SKIP_DIRS = {"__pycache__", "node_modules", ".venv", "dist", "build", ".git"}
MAX_BYTES = 1_000_000

PATTERNS = {
    "planning folder": re.compile(r"\.planning/"),
    "approver": re.compile(r"master os-", re.IGNORECASE),
    # Four characters after "os-", at least one a digit, then a word end.
    "agent id": re.compile(r"os-(?=[a-z0-9]{4}(?![a-z0-9]))(?=[a-z]*[0-9])[a-z0-9]{4}"),
    "home path": re.compile(r"/Users/[a-z][a-z0-9_-]*/"),
}
# Real words that match a pattern above.
ALLOWED = (
    "/Users/alice/",
    "/Users/me/",
    "/Users/someone/",
    "/Users/user/",
    "/Users/username/",
    "/Users/you/",
)


def _files():
    for root in ROOTS:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            if SKIP_DIRS.intersection(path.parts):
                continue
            if path.stat().st_size > MAX_BYTES or path == Path(__file__).resolve():
                continue
            yield path


def test_no_workshop_references():
    hits = []
    for path in _files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for name, pattern in PATTERNS.items():
                match = pattern.search(line)
                if match and not any(word in line for word in ALLOWED):
                    hits.append(
                        f"{path.relative_to(REPO)}:{number}: {name}: {match.group(0)}"
                    )
    assert not hits, "\n".join(hits[:60])
