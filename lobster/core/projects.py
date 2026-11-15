"""Local projects: a folder with a Lobster workspace that has an identity.

A project is the folder the user works in. Its identity lives in
``.lobster_workspace/project.json`` and holds nothing else: no provider, no
model, no key. A project's provider, model and experts are the workspace's own
configuration files, read here through the existing config classes so every
setting keeps one home.

Standard library only at import time, so callers that answer quickly (a CLI
verb, an MCP overview) do not pay for the engine.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from lobster.core.runtime.workspace import (
    PROJECT_FILE_NAME,
    WORKSPACE_FOLDER_NAME,
    find_project_root,
)

logger = logging.getLogger(__name__)

PROJECT_SCHEMA_VERSION = "lobster.project/1"
CREATED_PATHS_FILE = "created_paths.json"
RUNS_FOLDER_NAME = "runs"
# What the MCP server leaves in a run folder it made (or an unfinished one beside it).
RUN_MARKER = "mcp_turns.json"
RUN_UNFINISHED_PREFIX = ".lobster_workspace.unfinished-"
LOCK_SEARCH_DEPTH = 4
REGISTRY_FILE = "projects.json"
NAME_MAX = 256

_IDENTITY_KEYS = frozenset(
    {"schema_version", "project_id", "name", "created_at", "cloud"}
)
# Whole words that mark a setting or a secret. Matched per word of a key name
# (split on case changes and non-letters), so ``api_key`` is refused and
# ``keywords`` is not.
_FORBIDDEN_WORDS = frozenset(
    {
        "key",
        "keys",
        "apikey",
        "secret",
        "secrets",
        "token",
        "tokens",
        "password",
        "passwords",
        "passwd",
        "credential",
        "credentials",
        "cred",
        "creds",
        "auth",
        "authorization",
        "bearer",
        "passphrase",
        "pwd",
        "provider",
        "model",
    }
)
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_CREATED_AT = re.compile(
    r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(Z|[+-]\d\d:\d\d)", re.ASCII
)

PathLike = Union[str, Path]


class ProjectError(Exception):
    """A project operation refused or failed; ``code`` is machine-readable.

    Codes: ``not_a_project``, ``invalid_project``, ``bad_argument``,
    ``nested_project``, ``unsafe_path``, ``in_use``.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Project:
    root: Path
    project_id: Optional[str]  # None only for the stand-in of a broken project.json
    name: str
    created_at: Optional[str]
    cloud: Optional[Dict[str, Any]] = None

    @property
    def workspace(self) -> Path:
        return self.root / WORKSPACE_FOLDER_NAME


@dataclass(frozen=True)
class ProjectSettings:
    """What the project's workspace files say; empty ``experts`` means no explicit list."""

    provider: Optional[str]
    model: Optional[str]
    experts: List[str]
    preset: Optional[str]


@dataclass(frozen=True)
class RegistryEntry:
    project_id: str
    name: str
    root: str
    missing: bool


# --- identity file -----------------------------------------------------------


def _abs(path: PathLike) -> Path:
    # absolute() keeps links as the user typed them; resolve() would not.
    return Path(os.path.abspath(path))


def _bad_names(data: Any, prefix: str = "") -> List[str]:
    found: List[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            words = {w.lower() for w in _WORD.findall(str(key))}
            if words & _FORBIDDEN_WORDS:
                found.append(f"{prefix}{key}")
            found.extend(_bad_names(value, f"{prefix}{key}."))
    elif isinstance(data, list):
        for item in data:
            found.extend(_bad_names(item, prefix))
    return found


def _clean(text: str) -> str:
    """Drop control, bidi and other format characters so a name cannot steer a terminal."""
    return "".join(
        c for c in text if unicodedata.category(c) not in ("Cc", "Cf", "Zl", "Zp")
    )


def _name_ok(name: Any) -> bool:
    return isinstance(name, str) and 0 < len(name.strip()) <= NAME_MAX


def _validate_name(name: Any) -> str:
    """A name to store: 1 to ``NAME_MAX`` characters, none of them control or format."""
    if not _name_ok(name):
        raise ProjectError(
            "bad_argument", f"A project name has 1 to {NAME_MAX} characters."
        )
    if _clean(name) != name:
        raise ProjectError(
            "bad_argument",
            "A project name cannot hold control or invisible characters.",
        )
    return name.strip()


def _check_identity(data: Dict[str, Any]) -> None:
    """Raise ``invalid_project`` unless ``data`` is a well-formed identity."""
    if data.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise ProjectError(
            "invalid_project",
            f"project.json is not a {PROJECT_SCHEMA_VERSION} file.",
        )
    project_id = data.get("project_id")
    try:
        parsed = uuid.UUID(project_id)
        valid_id = parsed.version == 4 and str(parsed) == project_id
    except (ValueError, AttributeError, TypeError):
        valid_id = False
    if not valid_id:
        raise ProjectError("invalid_project", "project.json has no valid project_id.")
    if not _name_ok(data.get("name")) or not _clean(data["name"]).strip():
        raise ProjectError("invalid_project", "project.json has no valid name.")
    created = data.get("created_at")
    if not isinstance(created, str) or not _CREATED_AT.fullmatch(created):
        raise ProjectError("invalid_project", "project.json has no valid created_at.")
    if data.get("cloud") is not None and not isinstance(data["cloud"], dict):
        raise ProjectError(
            "invalid_project", "project.json cloud must be null or an object."
        )


def write_project_file(workspace: Path, fields: Dict[str, Any]) -> None:
    """Write ``project.json`` atomically; refuse anything but the identity fields."""
    unknown = sorted(set(fields) - _IDENTITY_KEYS)
    if unknown or set(fields) != _IDENTITY_KEYS:
        raise ProjectError(
            "bad_argument",
            "project.json holds identity only; refused keys: "
            + (", ".join(unknown) or "(missing fields)"),
        )
    if bad := _bad_names(fields):
        raise ProjectError(
            "bad_argument",
            "project.json never holds settings or credentials: " + ", ".join(bad),
        )
    if fields["cloud"] is not None:
        raise ProjectError("bad_argument", "The cloud link is reserved and stays null.")
    _validate_name(fields["name"])
    _check_identity(fields)
    _atomic_write(workspace / PROJECT_FILE_NAME, fields)


def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_project(root: PathLike) -> Project:
    """Read the project in ``root``; ``not_a_project`` or ``invalid_project`` otherwise."""
    root = _abs(root)
    path = root / WORKSPACE_FOLDER_NAME / PROJECT_FILE_NAME
    if not path.is_file():
        raise ProjectError("not_a_project", f"No Lobster project in {root}.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProjectError("invalid_project", f"project.json cannot be read: {exc}")
    if not isinstance(data, dict):
        raise ProjectError("invalid_project", "project.json is not a JSON object.")
    if bad := _bad_names(data):
        raise ProjectError(
            "invalid_project",
            "project.json must hold identity only; it has credential-like keys: "
            + ", ".join(bad),
        )
    _check_identity(data)
    return Project(
        root=root,
        project_id=data["project_id"],
        name=_clean(data["name"]).strip(),
        created_at=data["created_at"],
        cloud=data.get("cloud"),
    )


def find_project(start: Optional[PathLike] = None) -> Optional[Project]:
    """The project that contains ``start`` (default: the current folder), or None.

    A project file that cannot be read counts as no project here; ``load_project``
    reports why.
    """
    root = find_project_root(start)
    if root is None:
        return None
    try:
        return load_project(root)
    except ProjectError:
        return None


def project_for_workspace(path: PathLike) -> Optional[Project]:
    """The project owning a workspace folder, or a project folder, or None."""
    path = _abs(path)
    root = path.parent if path.name == WORKSPACE_FOLDER_NAME else path
    try:
        return load_project(root)
    except ProjectError:
        return None


def enclosing_project(root: PathLike) -> Optional[Project]:
    """The project that already contains ``root`` (a folder that may not exist yet), or None.

    Looks from the folder's parent, so a legacy workspace in ``root`` itself does
    not hide a project above it.
    """
    start = _abs(root).parent
    while not start.exists() and start.parent != start:
        start = start.parent
    return find_project(start)


def _same(a: PathLike, b: PathLike) -> bool:
    """Whether two paths are the same file (links, ``..`` and letter case resolved)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def is_home_or_above(root: PathLike) -> bool:
    """True for the user's home folder, any folder that contains it, and ``/``."""
    home = Path(os.path.realpath(Path.home()))
    return any(_same(root, candidate) for candidate in (home, *home.parents))


def init_project(root: PathLike, name: Optional[str] = None) -> Tuple[Project, bool]:
    """Create the project in ``root`` or keep the one there; returns ``(project, created)``.

    A folder that already has a project keeps its identity (``name`` renames it).
    A workspace without ``project.json`` is adopted: its files are left as they are.
    """
    root = _abs(root)
    if name is not None:
        name = _validate_name(name)
    if is_home_or_above(root):
        raise ProjectError(
            "unsafe_path",
            f"{root} is your home folder or contains it; make a subfolder, "
            "e.g. lobster init ~/my-study",
        )
    workspace = root / WORKSPACE_FOLDER_NAME
    if workspace.is_symlink():
        raise ProjectError(
            "unsafe_path", f"{workspace} is a link; a project needs a real folder."
        )
    if (workspace / PROJECT_FILE_NAME).exists():
        project = load_project(root)
        if name and name != project.name:
            write_project_file(
                workspace,
                {
                    "schema_version": PROJECT_SCHEMA_VERSION,
                    "project_id": project.project_id,
                    "name": name,
                    "created_at": project.created_at,
                    "cloud": project.cloud,
                },
            )
            project = load_project(root)
        _register(project)
        return project, False
    if (outer := enclosing_project(root)) is not None:
        raise ProjectError(
            "nested_project",
            f"{root} is inside the project {outer.name} at {outer.root}; "
            "projects do not nest.",
        )
    chosen = name or (root.name or "project")[:NAME_MAX]
    workspace.mkdir(parents=True, exist_ok=True)
    write_project_file(
        workspace,
        {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "project_id": str(uuid.uuid4()),
            "name": chosen,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cloud": None,
        },
    )
    project = load_project(root)
    _register(project)
    return project, True


def project_to_dict(project: Project) -> Dict[str, Any]:
    """The ``project`` block of ``lobster.project/1`` (field names follow Cloud's)."""
    return {
        "project_id": project.project_id,
        "name": project.name,
        "status": "active",
        "org_id": None,
        "created_at": project.created_at,
    }


# --- settings and folders -----------------------------------------------------


def project_settings(project: Project) -> ProjectSettings:
    """Provider, model and experts as the project's workspace files hold them."""
    # Imported here: the config classes pull in pydantic, which the identity code does not need.
    from lobster.config.workspace_agent_config import WorkspaceAgentConfig
    from lobster.config.workspace_config import WorkspaceProviderConfig

    provider_cfg = WorkspaceProviderConfig.load(project.workspace)
    provider = provider_cfg.global_provider
    model = provider_cfg.get_model_for_provider(provider) if provider else None
    agent_cfg = WorkspaceAgentConfig.load(project.workspace)
    experts = list(agent_cfg.enabled_agents)
    if not experts and agent_cfg.preset:
        # The file keeps either the list or the preset; the preset stands for its agents.
        from lobster.config.agent_presets import expand_preset

        experts = list(expand_preset(agent_cfg.preset) or [])
    return ProjectSettings(
        provider=provider,
        model=model,
        experts=experts,
        preset=agent_cfg.preset,
    )


@dataclass(frozen=True)
class Effective:
    """A setting a run would use and where it comes from (never a key)."""

    value: str
    source: str  # "project", "global", "shell" or "default"
    where: Optional[str] = None  # the global file, when source is "global"


def effective_settings(
    workspace: Path,
) -> Tuple[Optional[Effective], Optional[Effective]]:
    """``(provider, model)`` a run in *workspace* resolves today, the way a run does.

    ``(None, None)`` when no provider is set anywhere (workspace file, global file,
    ``LOBSTER_LLM_PROVIDER``).
    """
    from lobster.config.global_config import GlobalProviderConfig
    from lobster.core.config_resolver import ConfigResolver, ConfigurationError

    resolver = ConfigResolver(workspace)

    def mark(source: str) -> Tuple[str, Optional[str]]:
        if source.startswith("workspace"):
            return "project", None
        if source.startswith("global"):
            return "global", str(GlobalProviderConfig.get_config_path())
        if source.startswith("environment"):
            return "shell", None
        return "default", None

    try:
        provider, how = resolver.resolve_provider()
    except ConfigurationError:
        return None, None
    kind, where = mark(how)
    model, model_how = resolver.resolve_model(provider=provider)
    model_kind, model_where = mark(model_how)
    return (
        Effective(provider, kind, where),
        Effective(model, model_kind, model_where) if model else None,
    )


def run_workspace(folder: Path) -> Path:
    """The workspace ``-w <folder>`` names: the nested ``.lobster_workspace`` of a run
    folder (or project), else the folder itself when it holds its own ``data``."""
    nested = Path(folder) / WORKSPACE_FOLDER_NAME
    if nested.is_dir() and not (Path(folder) / "data").is_dir():
        return nested
    return Path(folder)


def runs_dir(project: Project) -> Path:
    return project.root / RUNS_FOLDER_NAME


def list_runs(project: Project) -> List[str]:
    """Names of the folders directly under the project's runs folder, sorted."""
    folder = runs_dir(project)
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_dir())


def _inside(root: Path, path: Path) -> Optional[Path]:
    """``path`` relative to ``root`` when it is strictly inside and no part is a link."""
    path = _abs(path)
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    if not rel.parts or WORKSPACE_FOLDER_NAME == rel.parts[0]:
        return None
    probe = root
    for part in rel.parts:
        probe = probe / part
        if probe.is_symlink():
            return None
    return rel


_UNREADABLE_RECORD = "the record of run folders is unreadable; run folders are kept"


def _read_created(project: Project) -> Tuple[List[str], Optional[str]]:
    """Recorded paths and, when the record exists but cannot be used, a warning."""
    path = project.workspace / CREATED_PATHS_FILE
    if not path.is_symlink() and not path.exists():
        return [], None
    try:
        if path.is_symlink():
            raise ValueError("link")
        paths = json.loads(path.read_text(encoding="utf-8"))["paths"]
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError("shape")
        return paths, None
    except (OSError, ValueError, KeyError, TypeError):
        return [], _UNREADABLE_RECORD


def record_created_path(project: Project, path: PathLike) -> None:
    """Remember a folder Lobster made inside the project so ``delete`` can remove it."""
    rel = _inside(project.root, _abs(path))
    if rel is None or ".." in rel.parts:
        raise ProjectError("unsafe_path", f"{path} is not a folder inside the project.")
    entry = rel.as_posix()
    recorded, _ = _read_created(project)
    if entry not in recorded:
        recorded.append(entry)
        _write_record(project, recorded, _recorded_files(project))


def record_created_file(project: Project, name: str) -> None:
    """Remember a file ``init`` wrote in the project folder (today: ``.env``)."""
    files = _recorded_files(project)
    if name not in files:
        paths, _ = _read_created(project)
        _write_record(project, paths, files + [name])


def _write_record(project: Project, paths: List[str], files: List[str]) -> None:
    _atomic_write(
        project.workspace / CREATED_PATHS_FILE,
        {"schema_version": "lobster.created_paths/1", "paths": paths, "files": files},
    )


def _recorded_files(project: Project) -> List[str]:
    try:
        files = json.loads(
            (project.workspace / CREATED_PATHS_FILE).read_text(encoding="utf-8")
        )["files"]
        return [f for f in files if isinstance(f, str)]
    except (OSError, ValueError, KeyError, TypeError):
        return []


# --- registry -----------------------------------------------------------------


def registry_path() -> Path:
    from lobster.config import global_config

    return global_config.CONFIG_DIR / REGISTRY_FILE


def _read_registry() -> List[Dict[str, str]]:
    try:
        data = json.loads(registry_path().read_text(encoding="utf-8"))
        return [
            e
            for e in data.get("projects", [])
            if isinstance(e, dict)
            and all(isinstance(e.get(k), str) for k in ("project_id", "name", "root"))
        ]
    except (OSError, ValueError, AttributeError):
        return []


def _write_registry(entries: List[Dict[str, str]]) -> None:
    _atomic_write(registry_path(), {"schema_version": 1, "projects": entries})


def _register(project: Project) -> None:
    # The list is a convenience for `projects list`; a failure here never blocks the project.
    try:
        root = str(project.root)
        entries = [e for e in _read_registry() if e["root"] != root]
        entries.append(
            {"project_id": project.project_id, "name": project.name, "root": root}
        )
        _write_registry(entries)
    except OSError as exc:
        logger.debug("project registry not updated: %s", exc)


def forget_project(root: PathLike) -> bool:
    """Drop the registry entry for ``root``; True when there was one."""
    root = str(_abs(root))
    entries = _read_registry()
    kept = [e for e in entries if e["root"] != root]
    if len(kept) == len(entries):
        return False
    try:
        _write_registry(kept)
    except OSError as exc:
        logger.debug("project registry not updated: %s", exc)
    return True


def registered_projects() -> List[RegistryEntry]:
    """Projects this machine has initialised; a folder that is gone is ``missing``."""
    out = []
    for e in _read_registry():
        ws = Path(e["root"]) / WORKSPACE_FOLDER_NAME
        missing = ws.is_symlink() or not (ws / PROJECT_FILE_NAME).is_file()
        out.append(
            RegistryEntry(e["project_id"], _clean(e["name"]), e["root"], missing)
        )
    return out


# --- delete -------------------------------------------------------------------


@dataclass
class DeletePlan:
    project: Project
    paths: List[Path] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    kept: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    env_note: Optional[str] = None  # None: no .env; "key": holds an API key; "plain"
    files: int = 0
    size_bytes: int = 0


def _measure(path: Path) -> Tuple[int, int]:
    files = size = 0
    for base, dirs, names in os.walk(path):  # links to folders are not entered
        for n in names:
            try:
                size += os.lstat(os.path.join(base, n)).st_size
            except OSError:
                continue
            files += 1
    return files, size


def _mode(path: Path) -> Optional[int]:
    try:
        return os.lstat(path).st_mode
    except FileNotFoundError:
        return None


def _is_run_folder(path: Path, parts: Tuple[str, ...]) -> bool:
    """``runs/<name>`` that carries what the MCP server leaves in a folder it made.

    The marker must be a regular file reached without a link; an unreadable
    folder raises ``OSError`` for the caller to report.
    """
    if len(parts) != 2 or parts[0] != RUNS_FOLDER_NAME or parts[1] == ".git":
        return False
    workspace = path / WORKSPACE_FOLDER_NAME
    ws_mode = _mode(workspace)
    if ws_mode is not None and stat.S_ISDIR(ws_mode):
        marker = _mode(workspace / RUN_MARKER)
        if marker is not None and stat.S_ISREG(marker):
            return True
    with os.scandir(path) as entries:
        return any(
            e.name.startswith(RUN_UNFINISHED_PREFIX)
            and stat.S_ISDIR(os.lstat(e.path).st_mode)
            for e in entries
        )


def _live_holder_below(path: Path) -> Optional[Tuple[Path, int]]:
    """A folder at or below ``path`` (a few levels, links not followed) held by a live run."""
    from lobster.cli_internal.mcp_lock import holder

    base_depth = len(path.parts)
    for base, dirs, _ in os.walk(path):
        folder = Path(base)
        if (pid := holder(folder)) is not None:
            return folder, pid
        if len(folder.parts) - base_depth >= LOCK_SEARCH_DEPTH:
            dirs[:] = []
    return None


def plan_delete(root: PathLike) -> DeletePlan:
    """What ``delete_project`` would remove: only things Lobster made.

    Raises ``not_a_project`` for a folder Lobster did not create and
    ``unsafe_path`` when the project is reached through a link.
    """
    from lobster.cli_internal.mcp_lock import holder

    root = _abs(root)
    if root.is_symlink():
        raise ProjectError(
            "unsafe_path", f"{root} is a link; pass the real project folder."
        )
    workspace = root / WORKSPACE_FOLDER_NAME
    project_file = workspace / PROJECT_FILE_NAME
    if workspace.is_symlink() or project_file.is_symlink():
        raise ProjectError(
            "unsafe_path",
            f"{workspace} or its project.json is a link; Lobster does not follow links.",
        )
    warnings: List[str] = []
    try:
        project = load_project(root)
    except ProjectError as exc:
        if exc.code == "not_a_project":
            raise ProjectError(
                "not_a_project",
                f"Lobster did not create a project in {root} (no {WORKSPACE_FOLDER_NAME}/{PROJECT_FILE_NAME}).",
            )
        # An identity that cannot be read still marks a Lobster workspace: it can go.
        project = Project(
            root=root, project_id=None, name=root.name or "project", created_at=None
        )
        warnings.append(
            f"{exc.message} Only the workspace and its run folders are removed."
        )
    plan = DeletePlan(project=project)
    plan.warnings.extend(warnings)
    recorded, unreadable = _read_created(project)
    if unreadable:
        plan.warnings.append(unreadable)
    seen: set = set()
    for rel in recorded:
        target = _inside(root, root / rel)
        if target is None or ".." in Path(rel).parts:
            plan.skipped.append((rel, "not a plain folder inside the project"))
            continue
        path = root / target
        if not path.is_dir():
            continue
        key = target.as_posix().casefold()
        if key in seen:
            continue
        seen.add(key)
        try:
            is_run = _is_run_folder(path, target.parts)
        except OSError:
            plan.warnings.append(f"could not read {path}; kept")
            continue
        if not is_run:
            plan.skipped.append((rel, "not a run folder Lobster made"))
            continue
        if any(_same(path, planned) for planned in plan.paths):
            continue
        if (busy := _live_holder_below(path)) is not None:
            raise ProjectError("in_use", f"{busy[0]} is in use by process {busy[1]}.")
        plan.paths.append(path)
    if (pid := holder(root)) is not None:
        raise ProjectError("in_use", f"{root} is in use by process {pid}.")
    from lobster.core.catalog.layout import CATALOG_DIR, own_catalog_dir

    if (catalog := own_catalog_dir(root)) is not None:
        plan.paths.append(catalog)
    plan.paths.append(workspace)
    for path in plan.paths:
        files, size = _measure(path)
        plan.files += files
        plan.size_bytes += size
    plan.env_note = _env_note(project)
    for name in (".env", "data", RUNS_FOLDER_NAME, CATALOG_DIR):
        if (root / name).exists() and (root / name) not in plan.paths:
            plan.kept.append(name)
    return plan


_KEY_NAME = re.compile(
    r"^\s*(?:export\s+)?(?:[A-Za-z0-9_]*_)?"
    r"(?:API_KEY|ACCESS_KEY|TOKEN|SECRET|PASSWORD|PASSWD)\s*=",
    re.IGNORECASE | re.MULTILINE,
)


def _env_note(project: Project) -> Optional[str]:
    """``key`` when ``.env`` is one init wrote or names an API key, token, secret or password, else ``plain``."""
    env = project.root / ".env"
    if not env.is_file():
        return None
    if ".env" in _recorded_files(project):
        return "key"
    try:
        return "key" if _KEY_NAME.search(env.read_text(encoding="utf-8")) else "plain"
    except (OSError, ValueError):
        return "plain"


def delete_project(plan: DeletePlan) -> List[str]:
    """Remove the planned paths and the registry entry; returns error messages."""
    errors: List[str] = []
    for path in plan.paths:
        if path.is_symlink():  # changed since the plan; never follow it
            errors.append(f"{path}: is a link, left in place")
            continue
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass  # already gone is as good as removed
        except OSError as exc:
            errors.append(f"{path}: {exc.strerror or exc}")
    workspace = plan.project.workspace
    if not workspace.exists() and not workspace.is_symlink():
        forget_project(plan.project.root)
    return errors
