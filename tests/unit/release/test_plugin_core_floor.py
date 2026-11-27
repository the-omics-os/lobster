"""Each plugin imports only core modules that its lowest allowed core has.

A plugin names the core it runs on as ``lobster-ai~=X.Y``, so pip may install
it next to core X.Y.0. A plugin that imports a core module added after that
installs without complaint and fails on import. The core's files at the
floor come from a list taken from git at that release and checked in, never from
an installed wheel or the clone's history, so a shallow clone runs every row.
"""

import ast
import hashlib
import subprocess  # nosec B404
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import pytest
from packaging.requirements import Requirement

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
# Packages with no Python code under lobster/: nothing they import can be missing
NO_LOBSTER_CODE = {"lobster-ai-tui"}  # ships the Go binary
PLUGINS = sorted(
    p.parent
    for p in (ROOT / "packages").glob("*/pyproject.toml")
    if p.parent.name not in NO_LOBSTER_CODE
)
# Plugins that import a core module newer than the core they allow; strict, so
# each row fails as soon as its pin moves and must leave this set
PINNED_TOO_LOW: Set[str] = set()
# The core's files at each floor a plugin names: a checked-in list taken from
# git at that release (its header has the commit and the command), or None for
# the release being built, which this tree stands for. Replace None with a list
# once that release exists.
CORE_FILES = {"1.3.0": "core_1_3_0_files.txt", "1.3.1": None, "1.3.2": None}
# sha256 of each list's file lines, so a list is not edited by hand unnoticed
CORE_FILES_SHA256 = {
    "core_1_3_0_files.txt": "e10595b6781b0a60bac160492c36ad4e99091fdb7b3b4d02b2a199c0b1d45ea9"
}
# Import failures the plugin handles itself
_OPTIONAL = {"ImportError", "ModuleNotFoundError", "Exception"}


def _floor(plugin: Path) -> Optional[str]:
    """The lowest core version the plugin allows, as X.Y.Z; None without one."""
    project = tomllib.loads((plugin / "pyproject.toml").read_text())["project"]
    for line in project.get("dependencies", []):
        requirement = Requirement(line)
        if requirement.name != "lobster-ai":
            continue
        for spec in requirement.specifier:
            if spec.operator in ("~=", ">=", "=="):
                parts = spec.version.split(".")
                return ".".join((parts + ["0", "0"])[:3])
    return None


def _modules(paths) -> Set[str]:
    """The modules and packages of files under lobster/, by dotted name."""
    found = set()
    for path in paths:
        parts = Path(path).with_suffix("").parts
        if Path(path).suffix != ".py" or parts[0] != "lobster":
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        found.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
    return found


def _listed(name: str) -> List[str]:
    """The file lines of a checked-in list, without its # header."""
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line and not line.startswith("#")]


def _core_at(floor: str) -> Set[str]:
    """The core's modules at release *floor*."""
    if floor not in CORE_FILES:
        pytest.fail(f"no file list for core {floor}: add one to CORE_FILES")
    name = CORE_FILES[floor]
    return _core_now() if name is None else _modules(_listed(name))


@lru_cache(maxsize=1)
def _core_now() -> Set[str]:
    return _modules(p.relative_to(ROOT) for p in (ROOT / "lobster").rglob("*.py"))


def _optional(tree: ast.AST) -> Set[int]:
    """Nodes under a try that handles an import failure, or under TYPE_CHECKING."""
    skipped = set()
    for node in ast.walk(tree):
        handled = isinstance(node, ast.Try) and any(
            h.type is None
            or {getattr(t, "id", None) for t in getattr(h.type, "elts", [h.type])}
            & _OPTIONAL
            for h in node.handlers
        )
        typing_only = isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(
            node.test
        )
        if handled or typing_only:
            skipped.update(id(n) for part in node.body for n in ast.walk(part))
    return skipped


def _imports(plugin: Path) -> Iterator[Tuple[str, int, str]]:
    """(file, line, name) of each ``lobster.*`` name the plugin's code imports:
    the module, and each name taken from it, which may be a submodule."""
    for path in sorted((plugin / "lobster").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skipped = _optional(tree)
        for node in ast.walk(tree):
            if id(node) in skipped:
                continue
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
                names += [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            for name in names:
                if name == "lobster" or name.startswith("lobster."):
                    yield path.relative_to(plugin.parent).as_posix(), node.lineno, name


def _missing(plugin: Path, floor_modules: Set[str]) -> List[str]:
    own = _modules(p.relative_to(plugin) for p in (plugin / "lobster").rglob("*.py"))
    return [
        f"{path}:{line} imports {name}"
        for path, line, name in _imports(plugin)
        if name in _core_now() and name not in own and name not in floor_modules
    ]


@pytest.mark.parametrize(
    "plugin",
    [
        pytest.param(
            p,
            marks=pytest.mark.xfail(
                p.name in PINNED_TOO_LOW,
                reason="plugin pin to lobster-ai>=1.3.1: a release decision",
                strict=True,
            ),
        )
        for p in PLUGINS
    ],
    ids=[p.name for p in PLUGINS],
)
def test_a_plugin_imports_only_modules_its_floor_core_has(plugin):
    floor = _floor(plugin)
    assert floor, f"{plugin.name} names no lowest lobster-ai version"

    missing = _missing(plugin, _core_at(floor))

    assert (
        not missing
    ), f"{plugin.name} allows lobster-ai {floor}, which lacks:\n" + "\n".join(missing)


@pytest.mark.parametrize(
    "line, floor",
    [("lobster-ai~=1.3", "1.3.0"), ("lobster-ai>=1.2.4,<2", "1.2.4"), ("numpy", None)],
)
def test_the_floor_is_read_from_the_core_requirement(tmp_path, line, floor):
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "p"\ndependencies = ["{line}"]\n'
    )
    assert _floor(tmp_path) == floor


def test_module_names_come_from_files_under_lobster():
    assert _modules(["lobster/a/__init__.py", "lobster/a/b.py", "x/c.py"]) == {
        "lobster",
        "lobster.a",
        "lobster.a.b",
    }


def test_a_module_taken_by_name_counts_and_the_plugins_own_do_not(tmp_path):
    code = tmp_path / "lobster" / "services" / "analysis"
    code.mkdir(parents=True)
    (code / "bulk_rnaseq_service.py").write_text("")  # its own copy of a core path
    (code / "uses.py").write_text(
        "from lobster.services.analysis import mann_whitney, bulk_rnaseq_service\n"
        "try:\n"
        "    import lobster.services.analysis.mann_whitney\n"
        "except ImportError:\n"
        "    pass\n"
    )
    floor = _core_now() - {
        "lobster.services.analysis.mann_whitney",
        "lobster.services.analysis.bulk_rnaseq_service",
    }

    missing = _missing(tmp_path, floor)

    assert [m.rsplit(" ", 1)[1] for m in missing] == [
        "lobster.services.analysis.mann_whitney"
    ], missing


def test_each_core_file_list_is_unchanged_and_matches_its_release():
    for name, digest in CORE_FILES_SHA256.items():
        lines = _listed(name)
        joined = "".join(line + "\n" for line in lines)
        assert hashlib.sha256(joined.encode()).hexdigest() == digest, name
        assert all(
            line.startswith("lobster/") and line.endswith(".py") for line in lines
        )
        # where this clone has the release commit, the list is exactly its files
        header = (FIXTURES / name).read_text(encoding="utf-8").splitlines()[0]
        commit = header.split("commit ")[1].split()[0]
        found = subprocess.run(  # nosec B603 B607
            ["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", commit, "--"]
            + ["lobster"],
            capture_output=True,
            text=True,
        )
        if found.returncode == 0:
            assert [f for f in found.stdout.split() if f.endswith(".py")] == lines
    assert set(CORE_FILES_SHA256) == {n for n in CORE_FILES.values() if n}


def test_a_package_left_out_has_no_lobster_code_and_no_core_floor():
    for name in NO_LOBSTER_CODE:
        package = ROOT / "packages" / name
        assert not (package / "lobster").exists(), name
        assert _floor(package) is None, name
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                else:
                    continue
                assert not [
                    n for n in names if n == "lobster" or n.startswith("lobster.")
                ], path
