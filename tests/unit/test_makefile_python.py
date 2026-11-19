"""make install accepts the Pythons the package supports, and no older one."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_makefile_minimum_python_is_the_packages():
    pyproject = (ROOT / "pyproject.toml").read_text()
    makefile = (ROOT / "Makefile").read_text()
    major, minor = re.search(r'requires-python = ">=(\d+)\.(\d+)', pyproject).groups()
    older = f"{major}.{int(minor) - 1}"

    assert f"PYTHON_VERSION_MIN := {major}.{minor}" in makefile
    # the version gates and the interpreter candidates
    assert f"({major},{int(minor) - 1})" not in makefile
    assert f"python{older} " not in makefile
    assert f"Python {older}+" not in makefile
