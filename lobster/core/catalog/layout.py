"""Where a workspace's catalog lives, and what its file keys are relative to.

A plain workspace keeps its catalog inside itself. A Lobster project keeps one
catalog for all its runs: the project's own workspace and every
``runs/<name>/.lobster_workspace`` use the catalog in ``<project>/.lamindb``, and a
saved file's key is its path relative to the project folder. Two runs that both
save ``data/x.h5ad`` are then two files, not two versions of one.

A project catalog made before runs were recorded sits in
``<project>/.lobster_workspace/.lamindb`` (its storage root is that workspace). It
keeps serving the project's own workspace; the runs cannot use it (their files are
outside its storage), and ``Layout.legacy`` says so.

Standard library and ``lobster.core.projects`` only: nothing here imports LaminDB.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lobster.core.projects import RUNS_FOLDER_NAME, ProjectError, load_project
from lobster.core.runtime.workspace import WORKSPACE_FOLDER_NAME

CATALOG_DIR = ".lamindb"
CONFIG_FILE = "lobster.json"


@dataclass(frozen=True)
class Layout:
    workspace: Path  # the session workspace asked about
    home: Path  # holds the catalog lock and Lobster's hidden records
    root: Path  # LaminDB's storage root; file keys are relative to it
    project: Optional[Path]  # the project folder, when ``home`` belongs to one
    legacy: bool  # a project catalog whose storage root is the project's workspace

    @property
    def catalog_dir(self) -> Path:
        return self.root / CATALOG_DIR

    @property
    def config_file(self) -> Path:
        return self.catalog_dir / CONFIG_FILE

    @property
    def runs_recorded(self) -> Optional[bool]:
        """Whether a run of this project is recorded here; None outside a project."""
        if self.project is None:
            return None
        return not self.legacy

    @property
    def legacy_fix(self) -> Optional[str]:
        """The line that moves an old project catalog aside, or None."""
        if not self.legacy:
            return None
        old = shell_quote(self.catalog_dir)
        return f"mv {old} {shell_quote(str(self.catalog_dir) + '.before-runs')}"

    @property
    def runs_refusal(self) -> Optional[str]:
        """Why a run workspace cannot use this layout's catalog, or None."""
        if not self.legacy or self.workspace == self.home:
            return None
        return (
            "this project's catalog was made before its runs were recorded and "
            f"cannot hold them; to record them, move it aside ({self.legacy_fix}) "
            "and the next run starts a new one"
        )


def shell_quote(path: object) -> str:
    text = str(path)
    return text if re.fullmatch(r"[A-Za-z0-9_./:@%+=,-]+", text) else repr(text)


def instance_name(workspace: Path) -> str:
    """A stable, readable instance name for a catalog home."""
    workspace = Path(workspace).resolve()
    slug = re.sub(r"[^a-z0-9]+", "-", workspace.name.lower()).strip("-")[:30]
    digest = hashlib.sha256(str(workspace).encode()).hexdigest()[:6]
    return f"lobster-{slug or 'workspace'}-{digest}"


def _project_root(folder: Path) -> Optional[Path]:
    try:
        load_project(folder)
    except ProjectError:
        return None
    except OSError:
        return None
    return folder


def _owner(workspace: Path) -> Optional[Path]:
    """The project whose own workspace, or run workspace, ``workspace`` is."""
    if workspace.name != WORKSPACE_FOLDER_NAME:
        return None
    parent = workspace.parent
    if (root := _project_root(parent)) is not None:
        return root
    if parent.parent.name == RUNS_FOLDER_NAME:
        return _project_root(parent.parent.parent)
    return None


def catalog_layout(workspace: Path, *, instance: bool = False) -> Layout:
    """The layout for ``workspace``; ``instance``: records go to a named instance.

    A named instance has no catalog folder of its own here, so an old project
    catalog does not matter to it: only the keys are decided (relative to the
    project).
    """
    ws = Path(workspace).resolve()
    project = _owner(ws)
    if project is None:
        return Layout(ws, ws, ws, None, False)
    home = project / WORKSPACE_FOLDER_NAME
    if not instance and ws != home and (ws / CATALOG_DIR / CONFIG_FILE).is_file():
        return Layout(ws, ws, ws, None, False)  # a run that has a catalog of its own
    made_before = (home / CATALOG_DIR / CONFIG_FILE).is_file()
    legacy = (
        not instance
        and made_before
        and not (project / CATALOG_DIR / CONFIG_FILE).is_file()
    )
    return Layout(ws, home, home if legacy else project, project, legacy)


_OWN_NAME = re.compile(r"lobster-[a-z0-9-]*-[0-9a-f]{6}")


def own_catalog_dir(project: Path) -> Optional[Path]:
    """``<project>/.lamindb`` when it is the catalog Lobster made for the project.

    Lobster's own mark (``lobster.json`` made by it, an instance name it gives) and
    LaminDB's (``storage_uid.txt`` of the same instance) must agree: a folder that
    only happens to be called ``.lamindb`` is the user's and is not ours to remove.
    """
    folder = Path(project) / CATALOG_DIR
    if folder.is_symlink() or not folder.is_dir():
        return None
    try:
        config = json.loads((folder / CONFIG_FILE).read_text(encoding="utf-8"))
        uid = (folder / "storage_uid.txt").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    slug = config.get("slug") if isinstance(config, dict) else None
    if not isinstance(slug, str) or config.get("created_by") != "lobster":
        return None
    if not _OWN_NAME.fullmatch(slug.rpartition("/")[2]):
        return None
    marks = {line.strip() for line in uid.splitlines()}
    return folder if f"instance_slug={slug}" in marks else None
