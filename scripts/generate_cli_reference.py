#!/usr/bin/env python
"""Write the CLI reference for the lobster-dev skill from ``lobster.commands/1``.

Usage: generate_cli_reference.py [--check]   (--check exits 1 when the file is stale)
"""

import os
import sys
from pathlib import Path

os.environ["LOBSTER_LANG"] = "en"  # English source strings, whatever the host locale

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "lobster/skills/lobster-dev/references/cli-reference.generated.md"
MIRROR = REPO / "skills/lobster-dev/references/cli-reference.generated.md"
HEADER = (
    "<!-- Generated from lobster.commands/1 by scripts/generate_cli_reference.py; "
    "do not edit -->"
)


def _arg_line(arg: dict) -> str:
    kind = arg["type"]
    if arg.get("choices"):
        kind += ": " + "|".join(arg["choices"])
    flags = "required" if arg["required"] else "optional"
    help_ = f" - {arg['help']}" if arg.get("help") else ""
    return f"  - `{arg['name']}` ({kind}, {flags}){help_}"


def render() -> str:
    from lobster.cli import app
    from lobster.cli_internal.command_registry import build_manifest

    commands = build_manifest(app)["commands"]
    out = [HEADER, "", "# Lobster CLI reference (generated)", ""]
    for title, surface, prefix in (
        ("CLI commands", "cli", "lobster "),
        ("Slash commands", "slash", ""),
    ):
        out += [f"## {title}", ""]
        for cmd in (c for c in commands if surface in c["surfaces"]):
            out.append(f"### `{prefix}{cmd['name']}`")
            out.append("")
            out.append(cmd["help"] or "(no help text)")
            if cmd["args"]:
                out.append("")
                out.extend(_arg_line(a) for a in cmd["args"])
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(argv: list) -> int:
    text = render()
    if "--check" in argv:
        stale = [
            p
            for p in (TARGET, MIRROR)
            if not p.exists() or p.read_text(encoding="utf-8") != text
        ]
        for p in stale:
            print(f"stale: {p.relative_to(REPO)}", file=sys.stderr)
        return 1 if stale else 0
    for p in (TARGET, MIRROR):
        p.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
