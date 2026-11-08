"""sgkit imports without the pkg_resources deprecation warning, and without
pkg_resources at all (setuptools >= 82). Each case runs in a fresh interpreter
because sgkit can only be imported once per process."""

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("sgkit")

BLOCK_PKG_RESOURCES = """
import importlib.abc, sys

class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "pkg_resources":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, _Block())
"""


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_sgkit_imports_without_pkg_resources():
    proc = _run(BLOCK_PKG_RESOURCES + """
import importlib.metadata
from lobster.services.analysis import gwas_service

assert gwas_service.SGKIT_AVAILABLE, "sgkit import failed"
assert gwas_service.sg.__version__ == importlib.metadata.version("sgkit")
assert "pkg_resources" not in sys.modules, "stand-in leaked"
print("ok")
""")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("ok")


def test_sgkit_import_does_not_emit_the_pkg_resources_warning():
    proc = _run("""
import warnings
warnings.filterwarnings("error", message="pkg_resources is deprecated")
from lobster.services.analysis import gwas_service

assert gwas_service.SGKIT_AVAILABLE, "sgkit import failed"
print("ok")
""")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("ok")


def test_stand_in_gap_degrades_gwas_instead_of_crashing(tmp_path):
    # An sgkit that needs a pkg_resources name the stand-in does not provide.
    fake = tmp_path / "sgkit"
    fake.mkdir()
    (fake / "__init__.py").write_text(
        "import pkg_resources\npkg_resources.require('sgkit')\n"
    )
    proc = _run(BLOCK_PKG_RESOURCES + f"""
sys.path.insert(0, {str(tmp_path)!r})
from lobster.services.analysis import gwas_service

assert not gwas_service.SGKIT_AVAILABLE
try:
    gwas_service._import_sgkit()
except ImportError as exc:
    assert exc.name == "pkg_resources", exc.name
    assert "pkg_resources.require" in str(exc), str(exc)
else:
    raise AssertionError("expected ImportError")
assert "pkg_resources" not in sys.modules, "stand-in leaked"
print("ok")
""")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("ok")
