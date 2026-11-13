"""``lobster lamin``: the optional LaminDB catalog of a workspace.

Registration is light: LaminDB is imported only inside a command, and only when
that command has to open the workspace catalog. A named instance is never
opened from here, because connecting to one registered on lamin.ai contacts
Lamin's servers.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import typer

from lobster.i18n import h

lamin_app = typer.Typer(
    name="lamin",
    help=h("Optional LaminDB catalog of this workspace"),
    no_args_is_help=True,
)

_WORKSPACE_HELP = "Workspace directory. Default: ./.lobster_workspace"
_JSON_HELP = "Output results as JSON (machine-readable)"
# LaminDB handles and instance names; nothing that .env would read differently.
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
_LAMIN_HOSTS = {"lamin.ai", "www.lamin.ai"}


def _workspace(path: Optional[str], *, create: bool = False) -> Path:
    from lobster.core.runtime.workspace import resolve_workspace

    return resolve_workspace(path, create=create)


def _engine_installed() -> bool:
    from lobster.core.catalog.factory import _load_engine

    return _load_engine() is not None


def _slug(workspace: Path) -> Optional[str]:
    from lobster.core.catalog.lamin import CATALOG_DIR, CONFIG_FILE, _read_json

    slug = _read_json(workspace / CATALOG_DIR / CONFIG_FILE).get("slug")
    return slug if isinstance(slug, str) else None


def _open(workspace: Path, *, create: bool = False, instance: Optional[str] = None):
    """The catalog engine (the workspace's, or ``instance``); LaminDB talks to stderr."""
    from lobster.core.catalog.lamin import LaminCatalog

    with contextlib.redirect_stdout(sys.stderr):
        return LaminCatalog(workspace, instance, create=create)


def _set_env(workspace: Path, key: str, value: Optional[str]) -> Path:
    from dotenv import dotenv_values, set_key, unset_key

    env_file = workspace / ".env"
    if value is None:
        if env_file.is_file() and key in dotenv_values(env_file):
            unset_key(str(env_file), key, quote_mode="never")
        return env_file
    env_file.parent.mkdir(parents=True, exist_ok=True)
    try:  # a new .env is private: it also holds provider keys
        os.close(os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    except FileExistsError:
        pass
    set_key(str(env_file), key, value, quote_mode="never")
    return env_file


# --- The question `lobster init` asks (Go wizard, questionary, classic) ---


def init_question() -> Tuple[str, str]:
    """Title and text of the init question; its default answer is No."""
    return h("LaminDB catalog"), h(
        "Record what Lobster does in a LaminDB catalog? Datasets saved in the "
        "workspace, sessions and exported notebooks are recorded with versions and "
        "hashes; `lobster lamin ls` lists them. The catalog lives in the workspace, "
        "works offline and contacts no server. Optional; needs lobster-ai[lamin]. "
        "If you later point Lobster at a LaminDB instance of your own (`lobster "
        "lamin use`), saved files are copied into that instance's storage (local "
        "storage only), and LaminDB contacts Lamin's servers (lamin.ai) only if the "
        "instance is registered there, or to look up an instance that is not set "
        "up on this machine."
    )


def saved_settings(workspace: Path) -> Dict[str, str]:
    """The catalog settings written in the workspace .env (only those present)."""
    from dotenv import dotenv_values

    from lobster.core.catalog.factory import ENV_ENABLE, ENV_INSTANCE

    env_file = workspace / ".env"
    values = dotenv_values(env_file) if env_file.is_file() else {}
    return {
        key: values[key]
        for key in (ENV_ENABLE, ENV_INSTANCE)
        if values.get(key) is not None
    }


def init_default(workspace: Optional[Path], *, global_config: bool) -> bool:
    """The answer the question starts at: the saved setting (shell ignored)."""
    from lobster.core.catalog.factory import parse_flag, resolve_catalog_config

    if global_config:
        from lobster.config.global_config import GlobalProviderConfig

        return bool(parse_flag(GlobalProviderConfig.load().lamin_enabled))
    return resolve_catalog_config(workspace or _workspace(None), environ={}).enabled


def apply_init_choice(
    choice: Optional[bool], workspace: Path, *, previous: Dict[str, str]
) -> None:
    """Save the answer in the workspace .env, after init has written its files.

    ``previous`` is ``saved_settings`` read before init wrote anything: what init's
    .env replaced is put back (the instance always, the switch when there is no
    answer). No is written only when something would otherwise switch the
    catalog on.
    """
    from lobster.core.catalog.factory import ENV_ENABLE, resolve_catalog_config

    now = saved_settings(workspace)
    for key, value in previous.items():
        if key not in now and (key != ENV_ENABLE or choice is None):
            _set_env(workspace, key, value)
    if choice is None:
        return
    if choice:
        _set_env(workspace, ENV_ENABLE, "1")
    elif (
        ENV_ENABLE in previous or resolve_catalog_config(workspace, environ={}).enabled
    ):
        _set_env(workspace, ENV_ENABLE, "0")


def init_notes(choice: Optional[bool], *, global_config: bool) -> list:
    """Lines `lobster init` prints after saving the answer."""
    if choice is None:
        return []
    if not choice:
        return [h("LaminDB catalog: off.")]
    notes = [
        (
            h("LaminDB catalog: on for every workspace without its own setting.")
            if global_config
            else h("LaminDB catalog: on. `lobster lamin status` shows it.")
        )
    ]
    if not _engine_installed():
        from lobster.core.component_registry import get_install_command

        notes.append(
            h("LaminDB is not installed yet. Install it with: {command}").format(
                command=get_install_command("lamin", is_extra=True)
            )
        )
    return notes


def _source_label(source: Optional[str]) -> str:
    labels = {
        "default": h("default"),
        "env": h("environment variable"),
        "workspace": h("workspace"),
        "global": h("global settings"),
    }
    return labels.get(source or "", source or "-")


def _status_label(status: Optional[str]) -> str:
    labels = {
        "scheduled": h("scheduled"),
        "restarted": h("restarted"),
        "started": h("started"),
        "completed": h("completed"),
        "errored": h("errored"),
        "aborted": h("aborted"),
    }
    return labels.get(status or "", status or "-")


def _print_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, default=str))


def _size(count: int) -> str:
    from lobster.cli_internal.commands.light.purge_commands import _format_size

    return _format_size(count)


def overview(workspace: Path) -> Dict[str, Any]:
    """What ``lobster lamin status`` reports, as plain data."""
    from lobster.core.catalog.factory import resolve_catalog_config
    from lobster.core.catalog.lamin import CATALOG_DIR

    config = resolve_catalog_config(workspace)
    info: Dict[str, Any] = {
        "workspace": str(workspace),
        "enabled": config.enabled,
        "enabled_source": config.source,
        "mode": config.mode,
        "instance": config.instance,
        "instance_source": config.instance_source,
        "engine_installed": _engine_installed(),
        "catalog_dir": str(workspace / CATALOG_DIR),
        "slug": _slug(workspace),
        "versions": {"files": 0, "bytes": 0},
        "contents": None,
        "problem": None,
    }
    if info["slug"]:
        info["versions"] = None  # not known until counted
    if info["slug"] and info["engine_installed"]:
        engine = None
        try:
            engine = _open(workspace)
            info["contents"] = engine.overview()
            info["location"] = str(engine.status().location)
        except Exception as exc:  # status never fails on a damaged catalog
            info["problem"] = f"{type(exc).__name__}: {exc}"[:300]
        if engine is not None:
            try:  # the files prune would remove: one function answers both
                with contextlib.redirect_stdout(sys.stderr):
                    kept = engine.kept_versions()
                info["versions"] = {
                    "files": len(kept),
                    "bytes": sum(row["bytes"] for row in kept),
                }
            except Exception:
                pass  # the size stays unknown; the rest of the status stands
    return info


@lamin_app.command(
    "status", help=h("Show whether this workspace is recorded, and where.")
)
def status_cmd(
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help=h(_WORKSPACE_HELP)
    ),
    as_json: bool = typer.Option(False, "--json", help=h(_JSON_HELP)),
):
    """Show whether this workspace is recorded, and where."""
    info = overview(_workspace(workspace))
    if as_json:
        _print_json(info)
        if info["problem"]:
            raise typer.Exit(1)
        return
    on = h("on") if info["enabled"] else h("off")
    typer.echo(
        h("Catalog: {state} (setting from {source})").format(
            state=on, source=_source_label(info["enabled_source"])
        )
    )
    if info["mode"] == "instance":
        typer.echo(
            h("Records go to the LaminDB instance {instance}.").format(
                instance=info["instance"]
            )
        )
        typer.echo(
            h(
                "Saved files are copied into that instance's storage (local storage "
                "only). LaminDB contacts Lamin's servers (lamin.ai) only if the "
                "instance is registered there, or to look up an instance that is not "
                "set up on this machine. Browse it with LaminDB's own tools."
            )
        )
    else:
        typer.echo(
            h("Records go to the workspace catalog in {path}.").format(
                path=info["catalog_dir"]
            )
        )
    if not info["engine_installed"]:
        typer.echo(
            h("LaminDB is not installed. Install the 'lamin' extra of lobster-ai.")
        )
    if info["slug"] is None:
        if info["mode"] != "instance":
            typer.echo(h("This workspace has no catalog yet."))
        return
    if info["mode"] == "instance":
        typer.echo(
            h("Workspace catalog (not in use): {slug}").format(slug=info["slug"])
        )
    else:
        typer.echo(h("Instance: {slug}").format(slug=info["slug"]))
    if info["versions"] is None:
        typer.echo(h("Earlier versions: size unknown."))
    elif info["versions"]["files"]:
        typer.echo(
            h(
                "Earlier versions kept: {size} (files: {files}). "
                "`lobster lamin prune` removes them."
            ).format(
                size=_size(info["versions"]["bytes"]), files=info["versions"]["files"]
            )
        )
    if info["problem"]:
        typer.echo(
            h("The catalog could not be opened: {problem}").format(
                problem=info["problem"]
            )
        )
        raise typer.Exit(1)
    contents = info["contents"]
    if contents is None:
        return
    typer.echo(
        h("Files: {artifacts}, sessions: {runs}, notebooks: {notebooks}").format(
            **contents
        )
    )
    last = contents["last_run"]
    if last:
        typer.echo(
            h("Last session: {name} ({status}, started {started_at})").format(
                name=last["name"] or "-",
                status=_status_label(last["status"]),
                started_at=last["started_at"] or "-",
            )
        )


@lamin_app.command("ls", help=h("List the files or sessions in the workspace catalog."))
def ls_cmd(
    runs: bool = typer.Option(False, "--runs", help=h("List sessions")),
    artifacts: bool = typer.Option(
        False, "--artifacts", help=h("List files (default)")
    ),
    limit: int = typer.Option(
        50, "--limit", "-n", min=1, help=h("Maximum number of rows")
    ),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help=h(_WORKSPACE_HELP)
    ),
    as_json: bool = typer.Option(False, "--json", help=h(_JSON_HELP)),
):
    """List the files or sessions in the workspace catalog."""
    if runs and artifacts:
        typer.echo(h("Choose --runs or --artifacts, not both."), err=True)
        raise typer.Exit(2)
    ws = _workspace(workspace)
    if _slug(ws) is None:
        typer.echo(h("This workspace has no catalog yet."), err=True)
        raise typer.Exit(1)
    if not _engine_installed():
        typer.echo(
            h("LaminDB is not installed. Install the 'lamin' extra of lobster-ai."),
            err=True,
        )
        raise typer.Exit(1)
    try:
        engine = _open(ws)
        rows = engine.list_runs(limit) if runs else engine.list_artifacts(limit)
    except Exception as exc:
        typer.echo(
            h("The catalog could not be opened: {problem}").format(
                problem=f"{type(exc).__name__}: {exc}"[:300]
            ),
            err=True,
        )
        raise typer.Exit(1)
    if as_json:
        _print_json(rows)
        return
    if not rows:
        typer.echo(h("Nothing recorded yet."))
        return
    for row in rows:
        if runs:
            counts = h("{outputs} out, {inputs} in").format(**row)
            recorded = [h("report")] if row.get("report") else []
            if row.get("environment"):
                recorded.append(h("environment"))
            notes = f"  [{', '.join(recorded)}]" if recorded else ""
            typer.echo(
                f"{row['started_at'] or '-':<19}  {_status_label(row['status']):<9}"
                f"  {row['name'] or '-'}  ({counts}){notes}"
            )
        else:
            mark = "*" if row["latest"] else " "
            notes = f"  [{', '.join(row['notes'])}]" if row["notes"] else ""
            kept = row["file"] or "-"
            where = (
                "" if kept == row["key"] else "  " + h("file: {path}").format(path=kept)
            )
            typer.echo(
                f"{row['created_at'] or '-':<19}  v{row['version']}{mark} "
                f" {row['key']}{where}{notes}"
            )


@lamin_app.command(
    "prune",
    help=h("Remove the kept copies of earlier versions; the records stay."),
)
def prune_cmd(
    dry_run: bool = typer.Option(
        False, "--dry-run", help=h("Show what would be removed; remove nothing")
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help=h("Do not ask first")),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help=h(_WORKSPACE_HELP)
    ),
    as_json: bool = typer.Option(False, "--json", help=h(_JSON_HELP)),
):
    """Remove the kept copies of earlier versions; the records stay."""
    ws = _workspace(workspace)
    if _slug(ws) is None:
        typer.echo(h("This workspace has no catalog yet."), err=True)
        raise typer.Exit(1)
    if not _engine_installed():
        typer.echo(
            h("LaminDB is not installed. Install the 'lamin' extra of lobster-ai."),
            err=True,
        )
        raise typer.Exit(1)

    def stop(message: str, exc: Exception):
        typer.echo(
            h(message).format(problem=f"{type(exc).__name__}: {exc}"[:300]), err=True
        )
        raise typer.Exit(1)

    try:
        engine = _open(ws)
        with contextlib.redirect_stdout(sys.stderr):
            versions = engine.kept_versions()
    except Exception as exc:
        stop("The catalog could not be opened: {problem}", exc)
    total = sum(row["bytes"] for row in versions)
    result: Dict[str, Any] = {
        "dry_run": dry_run,
        "files": len(versions),
        "bytes": total,
        "versions": versions,
        "removed": [],
        "failed": [],
        "skipped": [],
    }
    if versions and not dry_run:
        question = h(
            "Remove {size} of earlier versions (files: {files})? "
            "The records stay, marked as not kept."
        ).format(size=_size(total), files=len(versions))
        try:
            confirmed = yes or typer.confirm(question, err=True)
        except typer.Abort:
            confirmed = False
        if not confirmed:
            typer.echo(h("Nothing removed."), err=True)
            raise typer.Exit(1)
        from lobster.core.catalog.lamin import CatalogBusy, CatalogReadOnly

        try:
            with contextlib.redirect_stdout(sys.stderr):
                result.update(engine.prune_versions(versions))
        except CatalogBusy:
            typer.echo(
                h(
                    "Another process is using the catalog; nothing was removed. "
                    "Try again when it is idle."
                ),
                err=True,
            )
            raise typer.Exit(1)
        except CatalogReadOnly:
            typer.echo(
                h("The catalog cannot be written to; nothing was removed."), err=True
            )
            raise typer.Exit(1)
        except Exception as exc:
            stop("The catalog could not be pruned: {problem}", exc)
    if as_json:
        _print_json(result)
    elif not versions:
        typer.echo(h("No earlier versions to remove."))
    elif dry_run:
        for row in versions:
            typer.echo(
                f"{row['key']} v{row['version']}  {_size(row['bytes'])}  {row['file']}"
            )
        typer.echo(
            h("Would remove: {size} (files: {files}).").format(
                size=_size(total), files=len(versions)
            )
        )
    else:
        done = result["removed"]
        typer.echo(
            h(
                "Removed: {size} (files: {files}). "
                "The records stay, marked as not kept."
            ).format(size=_size(sum(r["bytes"] for r in done)), files=len(done))
        )
    for row in result["failed"] if not as_json else []:
        typer.echo(h("Could not remove {file}: {error}").format(**row), err=True)
    for row in result["skipped"] if not as_json else []:
        typer.echo(
            h("Not removed, it changed after it was listed: {file}").format(**row),
            err=True,
        )
    if result["failed"] or result["skipped"]:
        raise typer.Exit(1)


@lamin_app.command(
    "use", help=h("Record this workspace in a LaminDB instance of your own instead.")
)
def use_cmd(
    target: Optional[str] = typer.Argument(
        None,
        help=h("Instance as owner/name, or the storage folder of a local instance"),
    ),
    clear: bool = typer.Option(
        False, "--clear", help=h("Go back to the workspace catalog")
    ),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help=h(_WORKSPACE_HELP)
    ),
):
    """Record this workspace in a LaminDB instance of your own instead."""
    from lobster.core.catalog.factory import ENV_INSTANCE

    ws = _workspace(workspace, create=True)
    if clear or target is None:
        if not clear:
            typer.echo(h("Give an instance, or --clear."), err=True)
            raise typer.Exit(2)
        _set_env(ws, ENV_INSTANCE, None)
        typer.echo(h("This workspace records to its own catalog again."))
        return
    try:
        slug = instance_slug(target, ws)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2)
    env_file = _set_env(ws, ENV_INSTANCE, slug)
    typer.echo(
        h("This workspace now records to {slug} (saved in {path}).").format(
            slug=slug, path=env_file
        )
    )
    typer.echo(
        h(
            "Saved files are copied into that instance's storage (local storage "
            "only). LaminDB contacts Lamin's servers (lamin.ai) only if the "
            "instance is registered there, or to look up an instance that is not "
            "set up on this machine."
        )
    )


def instance_slug(target: str, workspace: Path) -> str:
    """``owner/name`` for a slug, a lamin.ai URL or a local instance's storage folder."""
    slug = _parse_instance(target, workspace)
    if not _SLUG.fullmatch(slug):
        raise ValueError(h("Give the instance as owner/name."))
    if slug == _slug(workspace):
        raise ValueError(h("That is this workspace; use --clear instead."))
    return slug


def _parse_instance(target: str, workspace: Path) -> str:
    from urllib.parse import urlsplit

    from lobster.core.catalog.lamin import CATALOG_DIR, CONFIG_FILE

    target = target.strip()
    path = Path(target).expanduser()
    if path.is_dir():
        path = path.resolve()
        if path == workspace.resolve():
            raise ValueError(h("That is this workspace; use --clear instead."))
        if (path / CATALOG_DIR / CONFIG_FILE).is_file():
            raise ValueError(
                h(
                    "That folder holds another workspace's catalog, whose settings "
                    "live in that workspace. Point at a LaminDB instance instead."
                )
            )
        mark = path / CATALOG_DIR / "storage_uid.txt"
        for line in mark.read_text().splitlines() if mark.is_file() else []:
            if line.startswith("instance_slug="):
                return line.split("=", 1)[1].strip()
        raise ValueError(
            h("No LaminDB instance stores its files in {path}.").format(path=path)
        )
    if "://" in target:
        url = urlsplit(target)
        if url.scheme not in ("http", "https") or url.hostname not in _LAMIN_HOSTS:
            raise ValueError(h("Give the instance as owner/name."))
        parts = [part for part in url.path.split("/") if part][:2]
    else:
        parts = target.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(h("Give the instance as owner/name."))
    return "/".join(parts)


@lamin_app.command(
    "init", help=h("Switch the catalog on for this workspace and create it now.")
)
def init_cmd(
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help=h(_WORKSPACE_HELP)
    ),
):
    """Switch the catalog on for this workspace and create it now."""
    from lobster.core.catalog.factory import (
        ENV_ENABLE,
        LOCK_FILE,
        resolve_catalog_config,
    )

    ws = _workspace(workspace, create=True)
    env_file = _set_env(ws, ENV_ENABLE, "1")
    config = resolve_catalog_config(ws)
    if not config.enabled:
        typer.echo(
            h(
                "Written to {path}, but the environment variable LOBSTER_LAMIN={value} "
                "keeps the catalog off until you unset it. Nothing was created."
            ).format(path=env_file, value=os.environ.get(ENV_ENABLE, ""))
        )
        return
    typer.echo(
        h("The catalog is on for this workspace (saved in {path}).").format(
            path=env_file
        )
    )
    if config.instance:
        typer.echo(
            h("Records go to the LaminDB instance {instance}.").format(
                instance=config.instance
            )
        )
        return
    if not _engine_installed():
        typer.echo(
            h("LaminDB is not installed. Install the 'lamin' extra of lobster-ai.")
        )
        return
    from lobster.core.queues.queue_storage import InterProcessFileLock

    try:
        with InterProcessFileLock(ws / LOCK_FILE):
            engine = _open(ws, create=True)
    except Exception as exc:
        typer.echo(
            h("The catalog could not be opened: {problem}").format(
                problem=f"{type(exc).__name__}: {exc}"[:300]
            ),
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(h("Workspace catalog ready: {slug}").format(slug=engine.slug))
