"""Lint the shipped agent skills: frontmatter, size ratchet, forbidden claims."""

import re
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
SKILLS = REPO / "lobster" / "skills"
SKILL_DIRS = sorted(p for p in SKILLS.iterdir() if (p / "SKILL.md").is_file())

# The target is 8 KB per SKILL.md body; both skills are under it, so the ceiling is
# the current size (a ratchet).
SKILL_MD_TARGET = 8 * 1024
SKILL_MD_CEILING = {"lobster-dev": 6762, "lobster-use": 7438}

# Phrases the project's claims rule forbids in shipped text. Case-insensitive regexes.
FORBIDDEN = [
    r"soc ?2",
    r"hipaa",
    r"[m2]fa enforced",
    r"alcoa",
    r"21 cfr part 11",
    r"gxp",
    r"tamper-evident",
    r"w3c-prov",
    r"sandboxed",
    r"state[- ]of[- ]the[- ]art",
    r"\bsota\b",
    r"guarantee",
    r"reproducib",  # flag only
    r"resumable",
]

# pre-existing wording, still to be reworked: (relative path, exact phrase as matched, lowercase)
ALLOWED_HITS = {
    ("lobster-dev/references/planning-workflow.md", "guarantee"),
    ("lobster-use/references/agent-patterns.md", "guarantee"),
    ("lobster-use/references/cli-reference.md", "guarantee"),
    ("lobster-use/references/cli-reference.md", "reproducib"),
}


def _frontmatter(skill: Path) -> dict:
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    return yaml.safe_load(text.split("---", 2)[1])


@pytest.mark.parametrize("skill", SKILL_DIRS, ids=lambda p: p.name)
def test_frontmatter(skill):
    meta = _frontmatter(skill)
    assert meta["name"] == skill.name
    assert len(meta["name"]) <= 64 and re.fullmatch(r"[a-z0-9-]+", meta["name"])
    assert 0 < len(str(meta.get("description") or "").strip()) <= 1024


def test_skill_md_size_ratchet():
    sizes = {p.name: (p / "SKILL.md").stat().st_size for p in SKILL_DIRS}
    print(f"SKILL.md bytes (target {SKILL_MD_TARGET}): {sizes}")
    for name, size in sizes.items():
        assert size <= SKILL_MD_CEILING.get(name, SKILL_MD_TARGET), (name, size)


def test_forbidden_claims():
    pattern = re.compile("|".join(FORBIDDEN), re.IGNORECASE)
    hits = {
        (str(p.relative_to(SKILLS)), m.group(0).lower())
        for p in SKILLS.rglob("*.md")
        for m in pattern.finditer(p.read_text(encoding="utf-8"))
    }
    assert hits - ALLOWED_HITS == set()


def test_generated_cli_reference_is_fresh():
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            str(REPO / "scripts" / "generate_cli_reference.py"),
            "--check",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
