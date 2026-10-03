"""Discover actual contract-marker expressions, not mentions inside strings."""

import argparse
import ast
from pathlib import Path


def has_contract_marker(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "contract"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "pytest"
        ):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    paths = [
        path
        for path in sorted(args.root.rglob("test_*.py"))
        if has_contract_marker(path.read_text())
    ]
    if not paths:
        parser.exit(1, "No contract marker expressions discovered\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
