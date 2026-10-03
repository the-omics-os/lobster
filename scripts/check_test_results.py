"""Reject empty, fully skipped, malformed, or failing JUnit test evidence."""

import argparse
import xml.etree.ElementTree as ET  # nosec B405 # Parser target below rejects every DOCTYPE before entity use.
from pathlib import Path


class _RejectDoctype(ET.TreeBuilder):
    """Reject DTD declarations through the parser, including encoded input."""

    def doctype(self, name, pubid, system):
        raise ValueError("DOCTYPE declarations are prohibited in JUnit evidence")


def check_result(path: Path, require_no_skips: bool = False) -> int:
    """Return executed test count; raise ValueError for unusable evidence."""
    parser = ET.XMLParser(
        target=_RejectDoctype()
    )  # nosec B314 # Parser target rejects DTDs/entities, including UTF-16 input.
    root = ET.parse(
        path, parser=parser
    ).getroot()  # nosec B314 # DOCTYPE-rejecting parser target; entities/DTDs cannot be used.
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError(f"{path}: no test cases")
    if any(
        case.find("failure") is not None or case.find("error") is not None
        for case in cases
    ):
        raise ValueError(f"{path}: failing test cases")
    # Also reject suite-level errors, which pytest can emit during collection.
    for suite in root.iter("testsuite"):
        if int(suite.get("failures", "0")) or int(suite.get("errors", "0")):
            raise ValueError(f"{path}: suite failures or collection errors")
    executed = sum(case.find("skipped") is None for case in cases)
    if not executed:
        raise ValueError(f"{path}: all test cases skipped")
    if require_no_skips and executed != len(cases):
        raise ValueError(f"{path}: required test cases skipped")
    return executed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--require-no-skips", action="store_true")
    args = parser.parse_args()
    for path in args.results:
        try:
            executed = check_result(path, require_no_skips=args.require_no_skips)
        except (OSError, ET.ParseError, ValueError) as exc:
            parser.exit(1, f"Invalid test evidence: {exc}\n")
        print(f"{path}: {executed} executed test cases")


if __name__ == "__main__":
    main()
