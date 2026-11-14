"""The ``lobster.commands/1`` manifest, derived from the Typer app.

The Typer app is the single source of truth: ``build_manifest`` walks it (and
the chat slash-command table) instead of keeping a second list of verbs.
Schema: ``lobster/schemas/lobster.commands.v1.schema.json``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import typer

from lobster.cli_internal.agent_stdio import AGENT_STDIO_COMMAND
from lobster.version import __version__

SCHEMA_VERSION = "lobster.commands/1"

# CLI verbs a scientist runs directly. Everything else (config, auth, packaging,
# maintenance, hidden protocol verbs) is setup or plumbing. Slash commands are
# all scientist-facing: they live inside the chat session.
SCIENTIST_FACING = frozenset(
    {
        "agents info",
        "agents list",
        "chat",
        "cloud chat",
        "cloud query",
        "dashboard",
        "init",
        "projects delete",
        "projects info",
        "projects list",
        "query",
        "status",
    }
)

_PY_TYPES = {bool: "boolean", int: "integer", float: "number"}
_CLICK_TYPES = {
    "boolean": "boolean",
    "integer": "integer",
    "integer range": "integer",
    "float": "number",
    "float range": "number",
    "path": "path",
}


def _first_line(text: Optional[str]) -> str:
    lines = (text or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def _arg_type(param: Any) -> Tuple[str, Optional[List[str]]]:
    desc = getattr(param, "type_descriptor", None)
    if desc is not None:
        # typer >= 0.29 vendors click and describes params through TypeDescriptor.
        if desc.is_choice:
            return "choice", [str(getattr(c, "value", c)) for c in desc.choices]
        if desc.is_path:
            return "path", None
        return _PY_TYPES.get(desc.annotation, "string"), None
    ptype = param.type  # classic click.ParamType
    if getattr(ptype, "choices", None):
        return "choice", [str(getattr(c, "value", c)) for c in ptype.choices]
    if getattr(param, "is_flag", False):
        return "boolean", None
    return _CLICK_TYPES.get(ptype.name, "string"), None


def _args(command: Any) -> List[Dict[str, Any]]:
    args = []
    for param in command.params:
        if getattr(param, "hidden", False):
            continue
        long_opts = [o for o in getattr(param, "opts", []) if o.startswith("--")]
        name = max(long_opts, key=len)[2:] if long_opts else param.name
        kind, choices = _arg_type(param)
        arg: Dict[str, Any] = {
            "name": name,
            "type": kind,
            "required": bool(param.required),
        }
        if choices is not None:
            arg["choices"] = choices
        if getattr(param, "help", None):
            arg["help"] = _first_line(param.help)
        args.append(arg)
    return args


def _cli_commands(group: Any, prefix: List[str]) -> List[Dict[str, Any]]:
    entries = []
    for name, command in group.commands.items():
        if command.hidden:
            continue
        path = prefix + [name]
        if hasattr(command, "commands"):
            entries.extend(_cli_commands(command, path))
            continue
        callback = command.callback
        full_name = " ".join(path)
        entries.append(
            {
                "name": full_name,
                "handler": f"{callback.__module__}:{callback.__qualname__}",
                "help": _first_line(command.help),
                "surfaces": ["cli"],
                "scientist_facing": full_name in SCIENTIST_FACING,
                "args": _args(command),
            }
        )
    return entries


def _slash_commands() -> List[Dict[str, Any]]:
    # Heavy module; imported only when a manifest is actually built.
    from lobster.cli_internal.commands.heavy.slash_commands import (
        extract_available_commands,
    )

    return [
        {
            "name": name,
            "handler": f"slash:{name}",
            "help": _first_line(description),
            "surfaces": ["slash"],
            "scientist_facing": True,
            "args": [],
        }
        for name, description in extract_available_commands().items()
    ]


def build_manifest(app: typer.Typer) -> Dict[str, Any]:
    """Return the ``lobster.commands/1`` manifest for ``app``, sorted by name."""
    commands = _cli_commands(typer.main.get_command(app), []) + _slash_commands()
    return {
        "schema_version": SCHEMA_VERSION,
        "engine_version": __version__,
        "agent_interface": AGENT_STDIO_COMMAND,
        "commands": sorted(commands, key=lambda c: c["name"]),
    }
