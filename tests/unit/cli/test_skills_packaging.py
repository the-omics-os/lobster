"""Agent skills ship as ``lobster/skills`` package data (sdist and wheel)."""

import filecmp
import subprocess  # nosec B404
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SKILL_FILES = [
    "lobster/skills/lobster-use/SKILL.md",
    "lobster/skills/lobster-dev/SKILL.md",
]


def _tree(root: Path) -> list:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_repo_root_skills_mirror_the_canonical_tree():
    # shortcut: repo-root skills/ is kept only because the published
    # skills.lobsterbio.com installer extracts lobster-main/skills/<name>/ from the
    # public tarball; drop this copy (and test) once the installer reads lobster/skills.
    canonical, mirror = REPO / "lobster" / "skills", REPO / "skills"
    names = [n for n in _tree(canonical) if "__pycache__" not in n]
    assert names == [n for n in _tree(mirror) if "__pycache__" not in n]
    _, mismatch, errors = filecmp.cmpfiles(canonical, mirror, names, shallow=False)
    assert not mismatch and not errors


@pytest.mark.slow
def test_sdist_and_wheel_carry_the_skills(tmp_path):
    pytest.importorskip("build")
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--skip-dependency-check",
            "--outdir",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    (sdist,) = tmp_path.glob("*.tar.gz")
    (wheel,) = tmp_path.glob("*.whl")
    with tarfile.open(sdist) as tar:
        prefix = tar.getnames()[0].split("/")[0]
        sdist_files = {
            m.name[len(prefix) + 1 :] for m in tar.getmembers() if m.isfile()
        }
    wheel_files = set(zipfile.ZipFile(wheel).namelist())
    for name in SKILL_FILES:
        assert name in wheel_files and name in sdist_files
    in_skills = {n for n in wheel_files if n.startswith("lobster/skills/")}
    assert in_skills == {n for n in sdist_files if n.startswith("lobster/skills/")}
