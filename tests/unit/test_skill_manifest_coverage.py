"""Check skill MANIFEST metadata without importing the engine or running installers."""

import unittest
from pathlib import Path, PurePosixPath

SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"
SKILL_NAMES = ("lobster-dev", "lobster-use")


def _manifest_entries(skill_dir: Path) -> list[str]:
    return [
        line.strip()
        for line in (skill_dir / "MANIFEST").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _markdown_paths(skill_dir: Path) -> set[str]:
    return {path.relative_to(skill_dir).as_posix() for path in skill_dir.rglob("*.md")}


def _missing_markdown(skill_dir: Path, entries: list[str]) -> set[str]:
    return _markdown_paths(skill_dir) - set(entries)


class TestSkillManifestCoverage(unittest.TestCase):
    def test_declared_paths_are_valid_and_unique(self):
        for name in SKILL_NAMES:
            skill_dir = SKILLS_ROOT / name
            entries = _manifest_entries(skill_dir)
            with self.subTest(skill=name):
                self.assertTrue(entries, "MANIFEST must not be empty")
                self.assertEqual(len(entries), len(set(entries)), "Duplicate entries")
            for entry in entries:
                with self.subTest(skill=name, entry=entry):
                    path = PurePosixPath(entry)
                    self.assertFalse(path.is_absolute(), "Use skill-relative paths")
                    self.assertNotIn("..", path.parts, "No parent traversal")
                    self.assertEqual(
                        entry, path.as_posix(), "Use canonical POSIX paths"
                    )
                    self.assertTrue(
                        (skill_dir / entry)
                        .resolve()
                        .is_relative_to(skill_dir.resolve()),
                        "Declared files must remain inside the skill directory",
                    )

    def test_declared_files_exist(self):
        for name in SKILL_NAMES:
            skill_dir = SKILLS_ROOT / name
            for entry in _manifest_entries(skill_dir):
                with self.subTest(skill=name, entry=entry):
                    self.assertTrue(
                        (skill_dir / entry).is_file(), "Missing declared file"
                    )

    def test_all_markdown_files_are_listed(self):
        for name in SKILL_NAMES:
            skill_dir = SKILLS_ROOT / name
            with self.subTest(skill=name):
                self.assertEqual(
                    _missing_markdown(skill_dir, _manifest_entries(skill_dir)),
                    set(),
                    "Markdown references must be listed in MANIFEST",
                )

    def test_missing_markdown_control_fires(self):
        skill_dir = SKILLS_ROOT / "lobster-dev"
        entries = _manifest_entries(skill_dir)
        omissions = {"references/aquadif-migration.md", "references/cli.md"}
        self.assertTrue(omissions.issubset(set(entries)))
        incomplete_entries = [entry for entry in entries if entry not in omissions]
        self.assertEqual(_missing_markdown(skill_dir, incomplete_entries), omissions)


if __name__ == "__main__":
    unittest.main()
