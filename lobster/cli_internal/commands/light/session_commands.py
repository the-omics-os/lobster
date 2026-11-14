"""``lobster sessions list``: the workspace's saved sessions (``--json`` for hosts)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from lobster.i18n import h

sessions_app = typer.Typer(
    name="sessions",
    help=h("Inspect saved sessions in a workspace."),
    no_args_is_help=True,
)

_PREVIEW_CHARS = 120


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


def _entry(workspace: Path, path: Path, session_id: str) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "session_id": session_id,
        "path": path.relative_to(workspace).as_posix(),
        "created": None,
        "updated": _iso(stat.st_mtime),
        "messages": None,
        "last_user_message": None,
        "_mtime": stat.st_mtime,
    }


def _read_session_file(workspace: Path, path: Path) -> Dict[str, Any]:
    entry = _entry(workspace, path, path.stem.removeprefix("session_"))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        conversation = data.get("conversation") or []
        entry["session_id"] = str(data.get("session_id") or entry["session_id"])
        created = (data.get("metadata") or {}).get("created_at")
        entry["created"] = None if created is None else str(created)
        entry["messages"] = len(conversation)
        user_messages = [
            m.get("content") for m in conversation if m.get("role") == "user"
        ]
        if user_messages:
            entry["last_user_message"] = str(user_messages[-1])[:_PREVIEW_CHARS]
    except Exception as exc:  # corrupt or foreign file: list it, don't crash
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def list_sessions(workspace: Path) -> List[Dict[str, Any]]:
    """Sessions found the way ``--session-id latest`` finds them, newest first.

    Transcript files ``session_*.json`` in the workspace root, plus session
    directories under ``.lobster/sessions`` that have no transcript yet.
    """
    entries = [
        _read_session_file(workspace, path) for path in workspace.glob("session_*.json")
    ]
    known = {entry["session_id"] for entry in entries}
    sessions_dir = workspace / ".lobster" / "sessions"
    if sessions_dir.is_dir():
        entries.extend(
            _entry(workspace, path, path.name)
            for path in sessions_dir.iterdir()
            if path.is_dir() and path.name not in known
        )
    entries.sort(key=lambda entry: entry["_mtime"], reverse=True)
    for entry in entries:
        del entry["_mtime"]
    return entries


@sessions_app.command(name="list")
def sessions_list(
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help=h(
            "Workspace directory. Defaults to ./.lobster_workspace or LOBSTER_WORKSPACE."
        ),
    ),
    as_json: bool = typer.Option(
        False, "--json", help=h("Print the lobster.sessions/1 listing as JSON.")
    ),
):
    from lobster.core.runtime.workspace import resolve_workspace

    workspace_path = resolve_workspace(explicit_path=workspace, create=False)
    sessions = list_sessions(workspace_path) if workspace_path.is_dir() else []
    if as_json:
        payload = {
            "schema_version": "lobster.sessions/1",
            "workspace": str(workspace_path),
            "sessions": sessions,
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return
    if not sessions:
        typer.echo(h("No sessions found in {path}").format(path=workspace_path))
        return
    for s in sessions:
        detail = s.get("error") or s["last_user_message"] or ""
        messages = "-" if s["messages"] is None else s["messages"]
        typer.echo(
            f"{s['session_id']:<36} {s['updated']:<20} {messages!s:>5}  {detail}"
        )


sessions_list.__doc__ = h("List saved sessions in a workspace, newest first.")
