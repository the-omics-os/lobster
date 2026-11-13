"""Which catalog a process gets.

Off unless the user switched it on. Resolution (first answer wins):

1. ``LOBSTER_LAMIN`` in the environment, as set in the shell (loading a ``.env``
   never puts it there, see ``Settings._load_credentials``),
2. ``LOBSTER_LAMIN`` in ``<workspace>/.env``,
3. ``lamin_enabled`` in the global ``providers.json``,
4. off.

The catalog location: ``LOBSTER_LAMIN_INSTANCE`` (environment, then workspace
``.env``), then ``LAMIN_CURRENT_INSTANCE``, else the workspace's own catalog. The
user's ambient Lamin default instance is never used.

One catalog per process: the engine's settings are process-wide, so a second
workspace in the same process gets a ``NullCatalog`` that says why.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple

from lobster.core.catalog.null import NullCatalog
from lobster.core.catalog.port import Catalog
from lobster.core.catalog.thread import ThreadedCatalog

logger = logging.getLogger(__name__)

ENV_ENABLE = "LOBSTER_LAMIN"
ENV_INSTANCE = "LOBSTER_LAMIN_INSTANCE"
ENV_LAMIN_INSTANCE = "LAMIN_CURRENT_INSTANCE"
GLOBAL_ENABLE_KEY = "lamin_enabled"
LOCK_FILE = Path(".lobster") / "catalog.lock"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class CatalogConfig:
    enabled: bool
    source: str  # env | workspace | global | default
    instance: Optional[str] = None  # None: the workspace's own catalog
    instance_source: Optional[str] = None

    @property
    def mode(self) -> str:
        return "instance" if self.instance else "workspace"


def parse_flag(value: Any) -> Optional[bool]:
    """``True``/``False`` for the usual spellings, ``None`` for unset or unknown."""
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def _read_workspace_env(workspace_path: Path) -> Mapping[str, Optional[str]]:
    env_file = Path(workspace_path) / ".env"
    if not env_file.is_file():
        return {}
    try:
        from dotenv import dotenv_values

        return dotenv_values(env_file)
    except Exception:
        logger.debug("Catalog: could not read %s", env_file, exc_info=True)
        return {}


def _global_config_path() -> Path:
    from lobster.config.global_config import CONFIG_DIR, CONFIG_FILE_NAME

    return CONFIG_DIR / CONFIG_FILE_NAME


def _read_global(path: Optional[Path]) -> Mapping[str, Any]:
    try:
        config_path = _global_config_path() if path is None else Path(path)
        if not config_path.is_file():
            return {}
        data = json.loads(config_path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.debug("Catalog: could not read the global config", exc_info=True)
        return {}


def _slug(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def resolve_catalog_config(
    workspace_path: Path,
    *,
    environ: Optional[Mapping[str, str]] = None,
    global_config_path: Optional[Path] = None,
) -> CatalogConfig:
    env = os.environ if environ is None else environ
    workspace_env = _read_workspace_env(workspace_path)

    enabled, source = False, "default"
    for name, mapping in (("env", env), ("workspace", workspace_env)):
        flag = parse_flag(mapping.get(ENV_ENABLE))
        if flag is not None:
            enabled, source = flag, name
            break
    else:
        flag = parse_flag(_read_global(global_config_path).get(GLOBAL_ENABLE_KEY))
        if flag is not None:
            enabled, source = flag, "global"

    instance, instance_source = None, None
    for name, value in (
        ("env", env.get(ENV_INSTANCE)),
        ("workspace", workspace_env.get(ENV_INSTANCE)),
        ("lamin_env", env.get(ENV_LAMIN_INSTANCE)),
    ):
        slug = _slug(value)
        if slug is not None:
            instance, instance_source = slug, name
            break

    return CatalogConfig(
        enabled=enabled,
        source=source,
        instance=instance,
        instance_source=instance_source,
    )


# The engine class, or None when it is not installed. Its constructor takes
# (workspace_path, instance=<slug or None>) and runs on the catalog worker thread.
EngineClass = Callable[..., Catalog]


def _load_engine() -> Optional[EngineClass]:
    if importlib.util.find_spec("lamindb_setup") is None:
        return None
    try:
        from lobster.core.catalog.lamin import (
            LaminCatalog,  # type: ignore[import-not-found]
        )
    except ImportError:
        return None
    return LaminCatalog


_process_lock = threading.Lock()
_process_catalog: Optional[Tuple[Path, Catalog]] = None
_hinted = False
# Settings already resolved to "off" in this process, keyed by workspace and the
# environment values that could change the answer: no file is read twice.
_resolved_off: set = set()


def _after_fork_in_child() -> None:
    global _process_lock
    _process_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def get_catalog(
    workspace_path: Path,
    *,
    environ: Optional[Mapping[str, str]] = None,
    global_config_path: Optional[Path] = None,
) -> Catalog:
    """The catalog for ``workspace_path`` in this process; never raises."""
    global _process_catalog
    try:
        env = os.environ if environ is None else environ
        # Explicitly off: no file is read.
        if parse_flag(env.get(ENV_ENABLE)) is False:
            return NullCatalog()
        workspace = Path(workspace_path).resolve()
        cache_key = None
        if environ is None and global_config_path is None:
            cache_key = (workspace, env.get(ENV_ENABLE), env.get(ENV_INSTANCE))
        with _process_lock:
            if _process_catalog is not None:
                bound, catalog = _process_catalog
                if bound == workspace:
                    return catalog
                return NullCatalog(
                    reason=f"this process already records to the catalog of {bound}"
                )
            if cache_key is not None and cache_key in _resolved_off:
                return NullCatalog()
            config = resolve_catalog_config(
                workspace, environ=environ, global_config_path=global_config_path
            )
            if not config.enabled:
                if cache_key is not None:
                    _resolved_off.add(cache_key)
                return NullCatalog()
            catalog = _build(workspace, config)
            if not isinstance(catalog, NullCatalog):
                _process_catalog = (workspace, catalog)
            return catalog
    except Exception:
        logger.debug("Catalog: selection failed; continuing without", exc_info=True)
        return NullCatalog(reason="the catalog settings could not be read")


def _build(workspace: Path, config: CatalogConfig) -> Catalog:
    global _hinted
    engine_class = _load_engine()
    if engine_class is None:
        if not _hinted:
            _hinted = True
            logger.warning(
                "The LaminDB catalog is switched on (setting from %s) but not "
                "installed; this session is not recorded. Install the 'lamin' extra "
                "of lobster-ai.",
                config.source,
            )
        return NullCatalog(
            reason="switched on, but the LaminDB engine is not installed"
        )
    instance = config.instance
    if instance is None:
        from lobster.core.catalog.lamin import workspace_settings_conflict

        conflict = workspace_settings_conflict(workspace)
        if conflict:
            logger.warning("The LaminDB catalog is off for this session: %s.", conflict)
            return NullCatalog(reason=conflict)
    else:
        logger.warning(
            "The LaminDB catalog records this session in the instance '%s' (setting "
            "from %s). Saved files are copied into that instance's storage (local "
            "storage only). LaminDB contacts Lamin's servers (lamin.ai) only if the "
            "instance is registered there, or to look up an instance that is not set "
            "up on this machine.",
            instance,
            config.instance_source or config.source,
        )
    return ThreadedCatalog(
        lambda: engine_class(workspace, instance=instance),
        lock_path=workspace / LOCK_FILE,
        engine="lamindb",
        mode=config.mode,
    )


def begin_session_run(
    data_manager: Any,
    workspace_path: Path,
    session_id: str,
    session_dir: Optional[Path] = None,
    kind: str = "session",
) -> Catalog:
    """Attach this process's catalog to ``data_manager`` and start a run; never raises.

    One run per client session (``AgentClient``) or per command invocation
    (``CommandClient``). Returns the catalog (``NullCatalog`` when off).
    """
    try:
        from lobster.core.catalog.port import SessionInfo
        from lobster.version import __version__

        catalog = get_catalog(workspace_path)
        attach = getattr(data_manager, "attach_catalog", None)
        if callable(attach):
            attach(catalog)
        if isinstance(catalog, NullCatalog):
            return catalog
        catalog.begin_run(
            SessionInfo(
                session_id=session_id,
                workspace_path=Path(workspace_path),
                lobster_version=__version__,
                session_dir=None if session_dir is None else Path(session_dir),
                kind=kind,
            )
        )
        return catalog
    except Exception:
        logger.debug("Catalog: session run not started", exc_info=True)
        return NullCatalog(reason="the session run could not be started")


def begin_command_run(
    data_manager: Any,
    workspace_path: Path,
    session_id: Optional[str],
    session_dir: Optional[Path] = None,
) -> Catalog:
    """``begin_session_run`` for one ``lobster command`` invocation; never raises."""
    run_id = session_id
    if not run_id or run_id == "command":
        run_id = datetime.now().strftime("command_%Y%m%d_%H%M%S")
    return begin_session_run(
        data_manager, workspace_path, run_id, session_dir, kind="command"
    )


def _reset_for_tests() -> None:
    """Forget the process catalog (tests only); closes it first."""
    global _process_catalog, _hinted
    with _process_lock:
        if _process_catalog is not None:
            catalog = _process_catalog[1]
            close = getattr(catalog, "close", None)
            if callable(close):
                close()
        _process_catalog = None
        _hinted = False
        _resolved_off.clear()
