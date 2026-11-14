#!/usr/bin/env python3
"""Validate NDJSON lines against a Lobster JSON schema.

Usage:
    lobster agent --stdio ... | python -I scripts/validate_ndjson.py lobster/schemas/protocol-v1.schema.json
    python -I scripts/validate_ndjson.py SCHEMA FILE.ndjson [FILE.ndjson ...]

Sibling schemas in the same directory resolve by file name ($ref). Exit code
0 when every line validates, 1 otherwise; each failure is printed as
``source:line: message``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List, TextIO, Tuple


def load_validator(schema_path: Path):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    resources = []
    for sibling in schema_path.parent.glob("*.json"):
        document = json.loads(sibling.read_text(encoding="utf-8"))
        resource = Resource.from_contents(document)
        resources.append((sibling.name, resource))
        if document.get("$id"):
            resources.append((document["$id"], resource))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    registry = Registry().with_resources(resources)
    return Draft202012Validator(schema, registry=registry)


def validate_lines(
    validator, source: str, lines: Iterable[str]
) -> List[Tuple[str, int, str]]:
    errors: List[Tuple[str, int, str]] = []
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append((source, number, f"not JSON: {exc}"))
            continue
        for error in validator.iter_errors(message):
            where = "/".join(str(p) for p in error.absolute_path) or "<root>"
            errors.append((source, number, f"{where}: {error.message}"))
    return errors


def main(argv: List[str], stdin: TextIO = sys.stdin) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    validator = load_validator(Path(argv[0]))
    sources = argv[1:]
    errors: List[Tuple[str, int, str]] = []
    if sources:
        for source in sources:
            with open(source, encoding="utf-8") as handle:
                errors.extend(validate_lines(validator, source, handle))
    else:
        errors.extend(validate_lines(validator, "<stdin>", stdin))
    for source, number, message in errors:
        print(f"{source}:{number}: {message}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
