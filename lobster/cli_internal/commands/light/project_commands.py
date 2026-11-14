"""``lobster projects list | info | delete``: local projects (``--json`` for hosts).

A project is a folder that ``lobster init`` set up. These verbs read and remove
that identity; they never create one (``lobster init`` does) and never touch a
file Lobster did not create.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional

import typer

from lobster.i18n import h

projects_app = typer.Typer(
    name="projects",
    help=h("List, inspect and delete local Lobster projects."),
    no_args_is_help=True,
)

SCHEMA_VERSION = "lobster.project/1"
_RUNS_LISTED = 100

# Error kinds follow the shared client vocabulary; exit 1 is a failure, 2 a usage error.
_KINDS = {
    "not_a_project": "not_found",
    "invalid_project": "invalid",
    "unsafe_path": "forbidden",
    "in_use": "conflict",
    "bad_argument": "usage",
    "nested_project": "usage",
    "needs_confirmation": "usage",
}
_ADVICE = {
    "not_a_project": "Run: lobster init [folder]",
    "in_use": "Stop the run that uses the folder, then try again.",
    "needs_confirmation": "Add --yes to delete, or --dry-run to list what would go.",
}


def _emit(payload: Dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def _fail(
    code: str,
    message: str,
    as_json: bool,
    exit_code: Optional[int] = None,
    advice: Optional[str] = None,
) -> NoReturn:
    kind = _KINDS.get(code, "internal")
    if as_json:
        _emit(
            {
                "error": message,
                "kind": kind,
                "status": None,
                "code": code,
                "retryable": False,
                "request_id": None,
            }
        )
    else:
        typer.echo(f"✗ {message}", err=True)
        if advice := advice or _ADVICE.get(code):
            typer.echo(f"  {advice}", err=True)
    raise typer.Exit(
        exit_code if exit_code is not None else (2 if kind == "usage" else 1)
    )


def _project_for(folder: Optional[Path], as_json: bool):
    """The project named by FOLDER, or the one that contains the current folder."""
    from lobster.core import projects
    from lobster.core.runtime.workspace import find_project_root

    try:
        if folder is not None:
            return projects.load_project(folder)
        root = find_project_root()
        if root is None:
            raise projects.ProjectError(
                "not_a_project",
                f"No Lobster project in {os.getcwd()} or above it.",
            )
        return projects.load_project(root)
    except projects.ProjectError as exc:
        _fail(exc.code, exc.message, as_json)


def _local_block(project) -> Dict[str, Any]:
    from lobster.core import projects

    settings = projects.project_settings(project)
    runs = projects.list_runs(project)
    return {
        "root": str(project.root),
        "workspace": str(project.workspace),
        "provider": settings.provider,
        "model": settings.model,
        "experts": settings.experts,
        "preset": settings.preset,
        "runs_folder": str(projects.runs_dir(project)),
        "runs": runs[:_RUNS_LISTED],
        "runs_total": len(runs),
    }


@projects_app.command(name="list")
def projects_list(
    as_json: bool = typer.Option(
        False, "--json", help=h("Print the lobster.project/1 listing as JSON")
    ),
):
    from lobster.core import projects

    entries = projects.registered_projects()
    if as_json:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "projects": [
                    {
                        "project_id": e.project_id,
                        "name": e.name,
                        "root": e.root,
                        "missing": e.missing,
                    }
                    for e in entries
                ],
                "total": len(entries),
            }
        )
        return
    if not entries:
        typer.echo(h("No projects yet. Run: lobster init [folder]"))
        return
    for e in entries:
        note = h("  (missing)") if e.missing else ""
        typer.echo(f"{e.name:<28} {e.project_id}  {e.root}{note}")


projects_list.__doc__ = h("List the projects this machine has set up.")


@projects_app.command(name="info")
def projects_info(
    folder: Optional[Path] = typer.Argument(
        None,
        help=h(
            "Project folder (default: the project that contains the current folder)"
        ),
        show_default=False,
    ),
    as_json: bool = typer.Option(
        False, "--json", help=h("Print the lobster.project/1 record as JSON")
    ),
):
    from lobster.core import projects

    project = _project_for(folder, as_json)
    local = _local_block(project)
    if as_json:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "project": projects.project_to_dict(project),
                "local": local,
            }
        )
        return
    typer.echo(f"{project.name}  ({project.project_id})")
    typer.echo(h("  folder:    {path}").format(path=project.root))
    typer.echo(h("  created:   {when}").format(when=project.created_at))
    typer.echo(
        h("  provider:  {value}").format(
            value=" / ".join(v for v in (local["provider"], local["model"]) if v)
            or h("not set")
        )
    )
    if not (local["provider"] and local["model"]):
        _echo_run_uses(project, local)
    experts = ", ".join(local["experts"]) or h("all installed")
    typer.echo(h("  experts:   {value}").format(value=experts))
    typer.echo(
        h("  runs:      {count} in {path}").format(
            count=local["runs_total"], path=local["runs_folder"]
        )
    )


projects_info.__doc__ = h(
    "Show a project's identity, provider, model, experts and runs."
)


def _echo_run_uses(project, local) -> None:
    """For what the project does not set: what a run uses, and from where."""
    from lobster.core import projects

    provider, model = projects.effective_settings(project.workspace)
    if provider is None:
        typer.echo(
            h("  a run uses: nothing yet; set one up with lobster init {path}").format(
                path=project.root
            )
        )
        return

    def where(setting) -> str:
        return {
            "global": setting.where,
            "shell": h("the shell"),
            "project": h("this project"),
            "default": h("provider default"),
        }[setting.source]

    parts = []
    if not local["provider"]:
        parts.append(
            h("provider {value} (from {where})").format(
                value=provider.value, where=where(provider)
            )
        )
    if not local["model"] and model is not None:
        parts.append(
            h("model {value} (from {where})").format(
                value=model.value, where=where(model)
            )
        )
    if parts:
        typer.echo(h("  a run uses: {parts}").format(parts=", ".join(parts)))


def _tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


@projects_app.command(name="delete")
def projects_delete(
    folder: Optional[Path] = typer.Argument(
        None,
        help=h("Project folder (default: the project in the current folder)"),
        show_default=False,
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help=h("Delete without asking (needed without a terminal)"),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help=h("List what would be removed; change nothing")
    ),
    as_json: bool = typer.Option(
        False, "--json", help=h("Print the result as lobster.project/1 JSON")
    ),
):
    from lobster.core import projects
    from lobster.core.runtime.workspace import find_project_root

    if folder is None:
        root = find_project_root()
        if root is None:
            _fail(
                "not_a_project",
                f"No Lobster project in {os.getcwd()} or above it.",
                as_json,
            )
        if Path(os.path.abspath(os.getcwd())) != root:
            # An enclosing project is easy to hit by accident; ask for it by name.
            _fail(
                "bad_argument",
                f"The project is {root}, not this folder. Pass its folder to delete it.",
                as_json,
            )
        folder = root
    try:
        plan = projects.plan_delete(folder)
    except projects.ProjectError as exc:
        if (
            exc.code == "not_a_project"
            and not dry_run
            and projects.forget_project(folder)
        ):
            message = h("Forgot the entry for {path}; no files were touched.").format(
                path=folder
            )
            if as_json:
                _emit(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "removed": [],
                        "kept": [],
                        "skipped": [],
                        "warnings": [],
                        "dry_run": False,
                        "deleted": True,
                    }
                )
            else:
                typer.echo(message)
            return
        _fail(
            exc.code,
            exc.message,
            as_json,
            advice=h("Nothing was changed.") if exc.code == "not_a_project" else None,
        )

    removed = [str(p) for p in plan.paths]
    kept = [str(plan.project.root / name) for name in plan.kept]
    warnings = list(plan.warnings)
    skipped = [{"path": rel, "reason": why} for rel, why in plan.skipped]
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project": projects.project_to_dict(plan.project),
        "removed": removed,
        "kept": kept,
        "skipped": skipped,
        "warnings": warnings,
        "dry_run": dry_run,
        "deleted": False,
    }
    if not as_json:
        typer.echo(
            h("Project {name} ({count} files, {size} bytes)").format(
                name=plan.project.name, count=plan.files, size=plan.size_bytes
            )
        )
        typer.echo(h("Lobster would remove:") if dry_run else h("Will remove:"))
        for path in removed:
            typer.echo(f"  {path}")
        for entry in plan.skipped:
            typer.echo(h("  kept {path}: {why}").format(path=entry[0], why=entry[1]))
        for warning in warnings:
            typer.echo(f"⚠ {warning}")
        typer.echo(
            h(
                "This removes Lobster's workspace, including the data and sessions saved in it."
            )
        )
        others = [p for p in kept if os.path.basename(p) != ".env"]
        if plan.env_note == "key":
            typer.echo(
                h(
                    ".env kept: it holds an API key; delete it yourself if you no longer need it"
                )
            )
        elif plan.env_note == "plain":
            typer.echo(h(".env kept"))
        if others:
            typer.echo(h("Left in place (not made by Lobster, or yours to keep):"))
            for path in others:
                typer.echo(f"  {path}")
    if dry_run:
        if as_json:
            _emit(result)
        return
    if not yes:
        if as_json or not _tty():
            _fail(
                "needs_confirmation",
                "Nothing changed: no terminal to ask on.",
                as_json,
                exit_code=2,
            )
        if not typer.confirm(
            h("Delete the workspace and everything saved in it?"), default=False
        ):
            typer.echo(h("Nothing deleted."))
            raise typer.Exit(1)
    errors: List[str] = projects.delete_project(plan)
    if errors:
        _fail("internal", "Could not remove everything: " + "; ".join(errors), as_json)
    result["deleted"] = True
    if as_json:
        _emit(result)
    else:
        typer.echo(
            h(
                "✓ Deleted the Lobster workspace and what was saved in it. The folder and the files outside the workspace are untouched."
            )
        )


projects_delete.__doc__ = h(
    "Delete a project's Lobster workspace; the folder and your other files stay."
)
