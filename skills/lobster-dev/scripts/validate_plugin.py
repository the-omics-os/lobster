#!/usr/bin/env python3
"""
Lobster Plugin Validator (standalone)

Zero-dependency validation script for Lobster AI agent packages.
Uses only Python 3.11+ stdlib (ast, re, tomllib, pathlib, sys, dataclasses).

Runs 8 structural checks against a plugin directory:
  1. PEP 420 compliance — no __init__.py at namespace boundaries
  2. Entry points — pyproject.toml has lobster.agents with :AGENT_CONFIG format
  3. AGENT_CONFIG position — appears before heavy imports (AST check)
  4. Factory signature — has 5 required parameters
  5. AQUADIF metadata — .metadata count matches @tool count
  6. Provenance calls — tools with provenance=True call log_tool_usage(ir=ir)
  7. Import boundaries — no cross-agent imports
  8. No ImportError guards — domain __init__.py has no try/except ImportError

SYNC NOTE: Built-in equivalent at lobster/scaffold/validators.py — keep checks aligned.

Usage:
    python scripts/validate_plugin.py ./lobster-epigenomics/
    python scripts/validate_plugin.py ../../packages/lobster-research/

Exit codes: 0 = all pass, 1 = any fail, 2 = usage error
"""

import ast
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ValidationResult:
    check: str
    passed: bool
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_agent_modules(plugin_dir: Path) -> List[Path]:
    """Find all Python files in lobster/agents/*/."""
    agents_dir = plugin_dir / "lobster" / "agents"
    if not agents_dir.exists():
        return []
    modules = []
    for d in agents_dir.iterdir():
        if d.is_dir() and d.name != "__pycache__":
            for py_file in d.glob("*.py"):
                if py_file.name != "__init__.py":
                    modules.append(py_file)
    return modules


def _has_provenance_call(tree: ast.AST) -> bool:
    """Check if AST contains log_tool_usage(ir=...) call.

    Inlined from lobster.config.aquadif.has_provenance_call to avoid imports.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "log_tool_usage":
                    for keyword in node.keywords:
                        if keyword.arg == "ir":
                            return True
    return False


# ---------------------------------------------------------------------------
# Check 1: PEP 420 compliance
# ---------------------------------------------------------------------------

def _check_pep420(plugin_dir: Path) -> List[ValidationResult]:
    violations = []
    for rel in ("lobster/__init__.py", "lobster/agents/__init__.py"):
        if (plugin_dir / rel).exists():
            violations.append(rel)

    if violations:
        return [ValidationResult(
            check="PEP 420 compliance",
            passed=False,
            message=f"Namespace boundary files exist (must be deleted): {', '.join(violations)}",
        )]
    return [ValidationResult(
        check="PEP 420 compliance",
        passed=True,
        message="No __init__.py at namespace boundaries",
    )]


# ---------------------------------------------------------------------------
# Check 2: Entry points
# ---------------------------------------------------------------------------

def _check_entry_points(plugin_dir: Path) -> List[ValidationResult]:
    pyproject = plugin_dir / "pyproject.toml"
    if not pyproject.exists():
        return [ValidationResult(
            check="Entry points",
            passed=False,
            message="pyproject.toml not found",
        )]

    try:
        parsed = tomllib.loads(pyproject.read_text())
    except Exception as e:
        return [ValidationResult(
            check="Entry points",
            passed=False,
            message=f"Failed to parse pyproject.toml: {e}",
        )]

    eps = parsed.get("project", {}).get("entry-points", {}).get("lobster.agents", {})
    if not eps:
        return [ValidationResult(
            check="Entry points",
            passed=False,
            message="No lobster.agents entry points found in pyproject.toml",
        )]

    results = []
    for name, value in eps.items():
        if not value.endswith(":AGENT_CONFIG"):
            results.append(ValidationResult(
                check="Entry points",
                passed=False,
                message=f"Entry point '{name}' value must end with ':AGENT_CONFIG', got '{value}'",
            ))
        else:
            results.append(ValidationResult(
                check="Entry points",
                passed=True,
                message=f"Entry point '{name}' = '{value}'",
            ))
    return results


# ---------------------------------------------------------------------------
# Check 3: AGENT_CONFIG position
# ---------------------------------------------------------------------------

_SKIP_FILES = {"shared_tools.py", "state.py", "config.py", "prompts.py", "conftest.py"}


def _check_agent_config_position(plugin_dir: Path) -> List[ValidationResult]:
    results = []
    for module_path in _find_agent_modules(plugin_dir):
        if module_path.name in _SKIP_FILES:
            continue

        try:
            tree = ast.parse(module_path.read_text())
        except SyntaxError as e:
            results.append(ValidationResult(
                check="AGENT_CONFIG position",
                passed=False,
                message=f"{module_path.name}: syntax error: {e}",
            ))
            continue

        config_line = None
        heavy_import_line = None

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "AGENT_CONFIG":
                        config_line = node.lineno
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", "") or ""
                names = [alias.name for alias in getattr(node, "names", [])]
                if any(m.startswith(("langgraph", "langchain")) for m in [mod] + names):
                    if heavy_import_line is None or node.lineno < heavy_import_line:
                        heavy_import_line = node.lineno

        if config_line is None:
            continue

        if heavy_import_line and config_line > heavy_import_line:
            results.append(ValidationResult(
                check="AGENT_CONFIG position",
                passed=False,
                message=f"{module_path.name}: AGENT_CONFIG at line {config_line} is after heavy import at line {heavy_import_line}",
            ))
        else:
            results.append(ValidationResult(
                check="AGENT_CONFIG position",
                passed=True,
                message=f"{module_path.name}: AGENT_CONFIG at line {config_line} (before heavy imports)",
            ))
    return results


# ---------------------------------------------------------------------------
# Check 4: Factory signature
# ---------------------------------------------------------------------------

_REQUIRED_PARAMS = {"data_manager", "callback_handler", "agent_name", "delegation_tools", "workspace_path"}


def _check_factory_signature(plugin_dir: Path) -> List[ValidationResult]:
    results = []
    for module_path in _find_agent_modules(plugin_dir):
        if module_path.name in _SKIP_FILES:
            continue

        try:
            tree = ast.parse(module_path.read_text())
        except SyntaxError:
            continue

        has_config = any(
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "AGENT_CONFIG" for t in node.targets)
            for node in ast.walk(tree)
        )
        if not has_config:
            continue

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                params = {arg.arg for arg in node.args.args}
                missing = _REQUIRED_PARAMS - params
                if not missing:
                    results.append(ValidationResult(
                        check="Factory signature",
                        passed=True,
                        message=f"{module_path.name}:{node.name}() has all standard params",
                    ))
                    break
                elif params & _REQUIRED_PARAMS:
                    results.append(ValidationResult(
                        check="Factory signature",
                        passed=False,
                        message=f"{module_path.name}:{node.name}() missing params: {sorted(missing)}",
                    ))
                    break
    return results


# ---------------------------------------------------------------------------
# Check 5: AQUADIF metadata
# ---------------------------------------------------------------------------

def _check_aquadif_metadata(plugin_dir: Path) -> List[ValidationResult]:
    results = []
    for module_path in _find_agent_modules(plugin_dir):
        if module_path.name != "shared_tools.py":
            continue

        source = module_path.read_text()
        metadata_count = len(re.findall(r'\w+\.metadata\s*=\s*\{', source))
        tool_count = source.count("@tool")

        if tool_count == 0:
            results.append(ValidationResult(
                check="AQUADIF metadata",
                passed=True,
                message=f"{module_path.name}: no tools defined (skeleton)",
            ))
        elif metadata_count >= tool_count:
            results.append(ValidationResult(
                check="AQUADIF metadata",
                passed=True,
                message=f"{module_path.name}: {metadata_count} metadata for {tool_count} tools",
            ))
        else:
            results.append(ValidationResult(
                check="AQUADIF metadata",
                passed=False,
                message=f"{module_path.name}: {metadata_count} metadata but {tool_count} @tool definitions",
            ))
    return results


# ---------------------------------------------------------------------------
# Check 6: Provenance calls
# ---------------------------------------------------------------------------

def _check_provenance_calls(plugin_dir: Path) -> List[ValidationResult]:
    results = []
    for module_path in _find_agent_modules(plugin_dir):
        if module_path.name != "shared_tools.py":
            continue

        try:
            source = module_path.read_text()
            tree = ast.parse(source)
        except SyntaxError as e:
            results.append(ValidationResult(
                check="Provenance calls",
                passed=False,
                message=f"{module_path.name}: syntax error: {e}",
            ))
            continue

        has_prov_true = '"provenance": True' in source or "'provenance': True" in source
        if has_prov_true:
            if _has_provenance_call(tree):
                results.append(ValidationResult(
                    check="Provenance calls",
                    passed=True,
                    message=f"{module_path.name}: log_tool_usage(ir=ir) calls found",
                ))
            else:
                results.append(ValidationResult(
                    check="Provenance calls",
                    passed=False,
                    message=f"{module_path.name}: provenance=True but no log_tool_usage(ir=ir) found",
                ))
        else:
            results.append(ValidationResult(
                check="Provenance calls",
                passed=True,
                message=f"{module_path.name}: no provenance-requiring tools",
            ))
    return results


# ---------------------------------------------------------------------------
# Check 7: Import boundaries
# ---------------------------------------------------------------------------

_CORE_AGENT_PATTERN = re.compile(
    r'from\s+lobster\.agents\.'
    r'(?!__)'
    r'(\w+)'
    r'\.'
    r'(\w+)'
    r'\s+import'
)


def _check_import_boundaries(plugin_dir: Path) -> List[ValidationResult]:
    results = []
    for module_path in _find_agent_modules(plugin_dir):
        source = module_path.read_text()
        domain = module_path.parent.name

        violations = []
        for match in _CORE_AGENT_PATTERN.finditer(source):
            imported_domain = match.group(1)
            if imported_domain != domain:
                violations.append(f"lobster.agents.{imported_domain}.{match.group(2)}")

        if violations:
            results.append(ValidationResult(
                check="Import boundaries",
                passed=False,
                message=f"{module_path.name}: imports from core agents: {', '.join(violations)}",
            ))
        else:
            results.append(ValidationResult(
                check="Import boundaries",
                passed=True,
                message=f"{module_path.name}: no cross-agent imports",
            ))
    return results


# ---------------------------------------------------------------------------
# Check 8: No ImportError guards
# ---------------------------------------------------------------------------

def _check_no_import_error_guard(plugin_dir: Path) -> List[ValidationResult]:
    results = []
    agents_dir = plugin_dir / "lobster" / "agents"
    if not agents_dir.exists():
        return results

    for domain_dir in agents_dir.iterdir():
        if not domain_dir.is_dir() or domain_dir.name == "__pycache__":
            continue

        init_file = domain_dir / "__init__.py"
        if not init_file.exists():
            continue

        try:
            tree = ast.parse(init_file.read_text())
        except SyntaxError:
            continue

        found_violation = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                exc_type = node.type
                if exc_type is None:
                    results.append(ValidationResult(
                        check="No ImportError guard",
                        passed=False,
                        message=f"{domain_dir.name}/__init__.py: bare except handler (line {node.lineno})",
                    ))
                    found_violation = True
                    break
                elif isinstance(exc_type, ast.Name) and exc_type.id == "ImportError":
                    results.append(ValidationResult(
                        check="No ImportError guard",
                        passed=False,
                        message=f"{domain_dir.name}/__init__.py: except ImportError at line {node.lineno}",
                    ))
                    found_violation = True
                    break
                elif isinstance(exc_type, ast.Tuple):
                    for elt in exc_type.elts:
                        if isinstance(elt, ast.Name) and elt.id == "ImportError":
                            results.append(ValidationResult(
                                check="No ImportError guard",
                                passed=False,
                                message=f"{domain_dir.name}/__init__.py: except (..., ImportError) at line {node.lineno}",
                            ))
                            found_violation = True
                            break
                    if found_violation:
                        break

        if not found_violation:
            results.append(ValidationResult(
                check="No ImportError guard",
                passed=True,
                message=f"{domain_dir.name}/__init__.py: clean (no ImportError guards)",
            ))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_plugin(plugin_dir: Path) -> List[ValidationResult]:
    """Run all 8 checks against a plugin directory."""
    results = []
    results.extend(_check_pep420(plugin_dir))
    results.extend(_check_entry_points(plugin_dir))
    results.extend(_check_agent_config_position(plugin_dir))
    results.extend(_check_factory_signature(plugin_dir))
    results.extend(_check_aquadif_metadata(plugin_dir))
    results.extend(_check_provenance_calls(plugin_dir))
    results.extend(_check_import_boundaries(plugin_dir))
    results.extend(_check_no_import_error_guard(plugin_dir))
    return results


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python validate_plugin.py <plugin-directory>", file=sys.stderr)
        return 2

    plugin_dir = Path(sys.argv[1]).resolve()
    if not plugin_dir.is_dir():
        print(f"Error: {plugin_dir} is not a directory", file=sys.stderr)
        return 2

    print("Lobster Plugin Validator (standalone)")
    print("=" * 37)
    print(f"Target: {plugin_dir}\n")

    results = validate_plugin(plugin_dir)

    if not results:
        print("[WARN] No checks produced results — is this a valid plugin directory?")
        print(f"       Expected structure: {plugin_dir}/lobster/agents/<domain>/")
        return 1

    passed = 0
    failed = 0
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        print(f"[{tag}] {r.check} — {r.message}")
        if r.passed:
            passed += 1
        else:
            failed += 1

    print(f"\nResult: {passed}/{passed + failed} PASS, {failed} FAIL")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
