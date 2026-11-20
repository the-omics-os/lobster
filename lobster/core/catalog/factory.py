"""Which catalog a process gets.

Off unless the user switched it on. Resolution (first answer wins):

1. ``LOBSTER_LAMIN`` in the environment, as set in the shell (loading a ``.env``
   never puts it there, see ``Settings._load_credentials``),
2. ``LOBSTER_LAMIN`` in ``<workspace>/.env``, then, for a run of a project, in
   the project's own workspace ``.env``,
3. ``lamin_enabled`` in the global ``providers.json``,
4. off.

The catalog location: ``LOBSTER_LAMIN_INSTANCE`` (environment, then workspace
``.env``), then ``LAMIN_CURRENT_INSTANCE``, else the workspace's own catalog. The
user's ambient Lamin default instance is never used.

One catalog per process: the engine's settings are process-wide, so a second
workspace in the same process gets a ``NullCatalog`` that says why. The runs of one
project are not second workspaces: they share the project's catalog (``layout``).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Tuple

from lobster.core.catalog.layout import Layout, catalog_layout
from lobster.core.catalog.null import NullCatalog
from lobster.core.catalog.port import Catalog
from lobster.core.catalog.thread import ThreadedCatalog, workspace_lock

logger = logging.getLogger(__name__)

ENV_ENABLE = "LOBSTER_LAMIN"
ENV_INSTANCE = "LOBSTER_LAMIN_INSTANCE"
ENV_LAMIN_INSTANCE = "LAMIN_CURRENT_INSTANCE"
GLOBAL_ENABLE_KEY = "lamin_enabled"
LOCK_FILE = Path(".lobster") / "catalog.lock"
# How long the first start of a catalog waits for another process that is creating it.
CREATE_WAIT_S = 30.0

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class CatalogConfig:
    enabled: bool
    source: str  # env | workspace | project | global | default
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
        data = json.loads(config_path.read_text(encoding="utf-8-sig"))
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
    layout = catalog_layout(Path(workspace_path))
    project_env = (
        _read_workspace_env(layout.home)
        if layout.project is not None and layout.workspace != layout.home
        else {}
    )

    enabled, source = False, "default"
    for name, mapping in (
        ("env", env),
        ("workspace", workspace_env),
        ("project", project_env),
    ):
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
        ("project", project_env.get(ENV_INSTANCE)),
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
        home = catalog_layout(workspace).home
        cache_key = None
        if environ is None and global_config_path is None:
            cache_key = (workspace, env.get(ENV_ENABLE), env.get(ENV_INSTANCE))
        with _process_lock:
            if _process_catalog is not None:
                bound, catalog = _process_catalog
                if bound == home:
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
                _process_catalog = (home, catalog)
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
    layout = catalog_layout(workspace, instance=instance is not None)
    if instance is None:
        from lobster.core.catalog.lamin import workspace_settings_conflict

        conflict = layout.runs_refusal or workspace_settings_conflict(workspace)
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
    lock_path = layout.home / LOCK_FILE
    return ThreadedCatalog(
        lambda: _start(engine_class, workspace, instance, layout, lock_path),
        lock_path=lock_path,
        engine="lamindb",
        mode=config.mode,
        key_root=layout.root,
    )


def _start(
    engine_class: EngineClass,
    workspace: Path,
    instance: Optional[str],
    layout: Layout,
    lock_path: Path,
) -> Catalog:
    """Build the engine. The first start creates the catalog under the lock, so two
    processes starting together (two servers on one project) do not both create it."""
    if instance is None and not layout.config_file.is_file():
        with workspace_lock(lock_path, CREATE_WAIT_S):
            return engine_class(workspace, instance=instance)
    return engine_class(workspace, instance=instance)


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
        abort_runs_on_sigterm()
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


def note_run_status(
    holder: Any,
    status: Optional[str],
    session_id: Optional[str] = None,
    reason: Optional[str] = None,
    *,
    if_unset: bool = False,
    if_unfinished: bool = False,
) -> None:
    """Say how the run of ``holder`` (a client or its data manager) ended; never raises.

    Every command sets it on each way out; a run nobody ended stays open, so a
    failed or killed command is never recorded as completed. A session clears
    it (None) as each turn starts: a turn cut short never reads as the last one.
    ``if_unset`` / ``if_unfinished``: see ``ThreadedCatalog.set_end_status``.
    """
    try:
        data_manager = getattr(holder, "data_manager", holder)
        setter = getattr(getattr(data_manager, "catalog", None), "set_end_status", None)
        if callable(setter):
            setter(
                status,
                session_id,
                reason,
                if_unset=if_unset,
                if_unfinished=if_unfinished,
            )
    except Exception:
        logger.debug("Catalog: run status not noted", exc_info=True)


# Exit codes of a process stopped by Ctrl-C (SIGINT) or SIGTERM.
_ABORT_EXIT_CODES = {130: "Ctrl-C", 143: "SIGTERM"}


def _exit_later(code: int) -> None:
    """End the process after the exit hook's budget, whatever is still running.

    A normal exit waits for every non-daemon thread (a tool still running in a
    pool); a SIGTERM must not wait for it.
    """
    from lobster.core.catalog import thread as catalog_thread

    timer = threading.Timer(
        catalog_thread.AT_EXIT_TIMEOUT_S + 1, _end_then_exit, (code,)
    )
    timer.daemon = True
    timer.start()


def _end_then_exit(code: int) -> None:
    try:
        _end_run_now()  # the exit may not have begun: a thread still holds it
    finally:
        os._exit(code)


_ending_early = False  # registered at most once per process


def end_run_when_exit_begins() -> None:
    """End the bound run as the exit begins, before Python waits for the threads
    still running: a tool stuck in a call would keep the run open for its whole
    length. The tool is not cut; the process still waits for it."""
    global _ending_early
    if _ending_early or _process_catalog is None:
        return
    try:
        threading._register_atexit(_exit_begins)
    except (AttributeError, RuntimeError):  # no such hook, or the exit already began
        logger.debug("Catalog: run not ended early", exc_info=True)
        return
    _ending_early = True


def _end_run_now() -> None:
    bound = _process_catalog
    end = getattr(None if bound is None else bound[1], "end_at_exit", None)
    if callable(end):
        end()


def _stop_now_text() -> str:
    from lobster.i18n import h

    return h(
        "Stopping without waiting for the running tool; a file it was writing may be incomplete."
    )


def _exit_begins() -> None:
    import atexit
    import signal

    _end_run_now()
    text = _stop_now_text()
    try:
        signal.signal(signal.SIGINT, lambda signum, frame: _stop_waiting(text))
        # the wait for threads ends where the atexit hooks begin: Ctrl-C is Python's
        atexit.register(signal.signal, signal.SIGINT, signal.default_int_handler)
    except (ValueError, OSError):
        logger.debug("Catalog: Ctrl-C at exit not handled", exc_info=True)


def _say(text: str) -> None:
    """A line from a signal handler; the interrupted thread may be writing stderr."""
    import sys

    try:
        print(text, file=sys.stderr, flush=True)
    except (RuntimeError, OSError):
        pass


def _stop_waiting(text: str) -> None:
    """A Ctrl-C while the exit waits for a running tool: stop waiting, as Python's
    own second Ctrl-C does, without its traceback."""
    import atexit
    import signal
    import sys

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the exit hooks run once
    _say(text)
    atexit._run_exitfuncs()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    os._exit(130)


def _on_sigterm(signum: int, frame: Any) -> None:
    import signal

    # No lock: the handler may run while this thread holds it.
    bound = _process_catalog
    catalog = None if bound is None else bound[1]
    setter = getattr(catalog, "set_end_status", None)
    if not callable(setter):
        # No run to end: die of the signal, as without this handler.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)
        return
    # A running turn is aborted; between a session's turns the last one stands.
    setter("aborted", None, "SIGTERM", if_unfinished=True)
    end_run_when_exit_begins()
    _exit_later(143)
    # Exiting (unlike dying of the signal) lets the exit hook end the run.
    raise SystemExit(143)


def abort_runs_on_sigterm() -> None:
    """A SIGTERM ends the open run as aborted; only where nobody handles it yet.

    Without a catalog run the handler lets the signal end the process as before.
    """
    import signal

    try:
        if threading.current_thread() is not threading.main_thread():
            return
        if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
            signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError, AttributeError):
        logger.debug("Catalog: SIGTERM not handled", exc_info=True)


def ctrl_c_ends_the_turn(holder: Any) -> Any:
    """A chat turn's Ctrl-C: the first ends the turn once its running step is done
    (the session's run goes on), the second stops at once."""
    return _ctrl_c_ends_the_run(holder, turn=True)


@contextmanager
def _ctrl_c_ends_the_run(holder: Any, *, turn: bool = False) -> Iterator[list]:
    """In a one-shot command, the first Ctrl-C ends the run at once and the second
    stops without waiting: a graph waits for its running tools before a Ctrl-C
    reaches the command, so the run cannot wait for the command to end it. In a
    chat ``turn`` the first only notes the turn aborted, should the second follow.

    Yields the presses seen; armed only on the main thread, with a bound run and
    Python's own Ctrl-C handler in place.
    """
    import signal

    pressed: list = []
    noting: list = []
    try:
        armed = (
            _process_catalog is not None
            and threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGINT) is signal.default_int_handler
        )
    except (ValueError, OSError, AttributeError):
        armed = False
    if not armed:
        yield pressed
        return
    from lobster.i18n import h

    waiting = h(
        "Stopping: waiting for the running step to finish; press Ctrl+C again to stop now."
    )
    stop_now = _stop_now_text()

    def first_press(signum: int, frame: Any) -> None:
        pressed.append(signum)
        signal.signal(signal.SIGINT, lambda s, f: _stop_waiting(stop_now))
        # not here: the handler may run while this thread holds the catalog's lock
        noted = threading.Thread(
            target=_abort_turn if turn else _abort_now,
            args=(holder,),
            name="lobster-ctrl-c",
            daemon=True,
        )
        noted.start()
        noting.append(noted)
        _say(waiting)
        signal.default_int_handler(signum, frame)

    signal.signal(signal.SIGINT, first_press)
    try:
        yield pressed
    finally:  # a caller that goes on after a press must not keep the hard exit
        signal.signal(signal.SIGINT, signal.default_int_handler)
        if turn:  # the turn's own end is noted after the press's
            for noted in noting:
                noted.join(timeout=1.0)


def _abort_turn(holder: Any) -> None:
    note_run_status(holder, "aborted", reason="Ctrl-C")


def _abort_now(holder: Any) -> None:
    note_run_status(holder, "aborted", reason="Ctrl-C")
    _end_run_now()


@contextmanager
def recorded_run_status(holder: Any, *, session: bool = False) -> Iterator[None]:
    """Note how the run of ``holder`` ended, on every way out of the block.

    Success (or an exit with code 0) is completed, Ctrl-C or SIGTERM is aborted,
    anything else (an error exit, an exception) is errored. A status noted in
    the block wins only over one read from an exit code: an exception or Ctrl-C
    always says how the run ended. For a chat (``session``) that ends normally,
    the status its last turn noted stands, and a turn cut short stays open.
    """
    with nullcontext([]) if session else _ctrl_c_ends_the_run(holder) as pressed:
        try:
            with _noted_run_status(holder, session=session):
                yield
        finally:
            if pressed:  # however the command went on, a Ctrl-C ended it
                note_run_status(holder, "aborted", reason="Ctrl-C")


@contextmanager
def _noted_run_status(holder: Any, *, session: bool) -> Iterator[None]:
    try:
        yield
    except KeyboardInterrupt:
        note_run_status(holder, "aborted", reason="Ctrl-C")
        end_run_when_exit_begins()
        raise
    except BaseException as exc:
        if _is_click_abort(exc):  # Ctrl-C at a click prompt
            note_run_status(holder, "aborted", reason="Ctrl-C")
            end_run_when_exit_begins()
            raise
        # typer.Exit carries exit_code, sys.exit code; any other error is its class.
        if isinstance(exc, SystemExit) or hasattr(exc, "exit_code"):
            code = exc.code if isinstance(exc, SystemExit) else exc.exit_code
            if code in (0, None):
                note_run_status(holder, "completed", if_unset=session)
            elif code in _ABORT_EXIT_CODES:
                reason = _ABORT_EXIT_CODES[code]
                note_run_status(holder, "aborted", reason=reason, if_unset=True)
                end_run_when_exit_begins()
            else:
                reason = f"exit code {code}"
                note_run_status(holder, "errored", reason=reason, if_unset=True)
        else:
            note_run_status(holder, "errored", reason=type(exc).__name__)
        raise
    note_run_status(holder, "completed", if_unset=session)


def _is_click_abort(exc: BaseException) -> bool:
    try:
        from click.exceptions import Abort
    except ImportError:
        return False
    return isinstance(exc, Abort)


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
