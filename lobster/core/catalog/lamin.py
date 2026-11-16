"""LaminDB engine for the workspace catalog (needs the ``lamin`` extra).

The only module that imports ``lamindb`` / ``lamindb_setup``, and only inside
``LaminCatalog.__init__``, which runs on the catalog worker thread.

Two modes:

- **Workspace catalog** (default when switched on): a local SQLite instance whose
  storage root is the workspace, so every file Lobster saves is registered where it
  was written (no copy) and nothing contacts a server. All Lamin state lives under
  ``<workspace>/.lamindb/``: Lamin's settings (and, unless the user set
  ``LAMIN_CACHE_DIR``, cache) directory is redirected there before
  ``lamindb_setup`` is imported, and init/connect run with ``_write_settings=False``,
  so the user's ``~/.lamin`` and the current directory are never touched. The
  instance slug is stored in ``<workspace>/.lamindb/lobster.json``. The redirect
  sets ``LAMIN_SETTINGS_DIR`` (and ``LAMIN_CACHE_DIR``) in ``os.environ`` for the
  rest of the process, so subprocesses Lobster starts, such as tool code, inherit
  them and a ``lamin`` command run from there sees the workspace catalog.
- **A named instance** (``LOBSTER_LAMIN_INSTANCE``): ``lamindb_setup.connect`` to
  it with the user's own Lamin settings (``_write_settings=False``). A saved file's
  bytes are copied into the instance's storage, at
  ``<storage root>/lobster/<workspace id>/<key>/<sha256><suffix>``, and that copy is
  registered where it is: bytes already recorded reuse their record and copy
  nothing (or put back a copy that left the storage). The workspace file is not changed.
  A run's environment (package names and versions) and step log are copied into the
  same storage, at ``<storage root>/lobster/<workspace id>/environments/`` and
  ``.../reports/``; ``INSTANCE_ENVIRONMENT`` below switches the environment off. Only an
  instance with local storage is supported. For a local instance this needs no
  server; LaminDB itself may contact Lamin's servers for a hub-registered one, and,
  as for any of its users, creates ``~/.lamin`` and may rewrite the instance's
  branch marker there. Two processes saving the same new bytes at once may each
  make a copy and a record; both hold those bytes.

The mapping (one Transform per Lobster version, one Run per session or command,
one Artifact version per explicit save, and per Run its environment, one hidden
``__lamindb_run__`` Artifact shared by every run with the same environment, and,
a snapshot of its step log as the run's report) is kept
in the ``_*`` helpers below so it can change without touching the port. A file
name always holds the latest bytes, as without a catalog: before a recorded,
unchanged file is overwritten, the saving thread keeps its bytes at
``<workspace>/.lamindb/versions/<key>/v<version><suffix>`` (the record's own version,
e.g. ``v0000.h5ad``; this engine only names the place) and the next registration
points its record there. One record per key is the latest: the one whose bytes the
file name holds.

Private LaminDB surfaces in use (pinned by tests): ``init(_write_settings=)``, ``connect(_write_settings=)``,
``InstanceSettings._persist``, ``instance_settings_file``, ``Run._status_code``,
``Artifact._real_key``, ``Artifact._key_is_virtual``, ``Artifact._storage_ongoing``,
``auto_storage_key_from_artifact``, ``hash_small_bytes``, the ``Run.environment``
and ``Run.report`` fields, the ``__lamindb_run__`` artifact kind and Django's
``Model._state.adding``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lobster.core.catalog.environment import environment_text
from lobster.core.catalog.layout import (  # noqa: F401 - the names callers use here
    CATALOG_DIR,
    CONFIG_FILE,
    catalog_layout,
    instance_name,
)
from lobster.core.catalog.port import (
    ArtifactRef,
    CatalogStatus,
    RunStatus,
    SessionInfo,
    TransformRef,
)

logger = logging.getLogger(__name__)

SETTINGS_SUBDIR = "settings"
CACHE_SUBDIR = "cache"
PIPELINE_KEY = "lobster-ai"
PIPELINE_REFERENCE = "https://github.com/the-omics-os/lobster"
VERSIONS_SUBDIR = "versions"
# LaminDB's own kind for what it records about a run (reports, environments).
# LaminDB hides it in ``to_dataframe`` only, not in ``filter()``, so this engine
# leaves it out of every lookup of a user's file itself (``_files``). Pinned by a
# test against LaminDB's ``ArtifactKind``.
RUN_ARTIFACT_KIND = "__lamindb_run__"
ENVIRONMENTS_SUBDIR = "environments"
REPORTS_SUBDIR = "reports"
# A step log above this size is not snapshotted: copying and hashing it has to fit
# in the exit budget with the rest of ``end_run``.
REPORT_MAX_BYTES = 8 * 1024 * 1024
# Decision 11 Nov 2026: secrets scan on, fail closed. A step log with a hit is not
# recorded (its file is never touched). Specific patterns, not entropy: the log
# carries 64-hex hashes. One entry per kind: (label, compiled pattern); the label is
# what the warning names, never the text that matched. Every pattern stays linear:
# bounded classes, no ``.*``.
#
# False positives: a key-name pattern needs the exact name, then a separator (``=`` or
# ``:``, after an optional closing quote) and at least 12 token characters; the
# prefixed forms need their exact prefix and a token of the stated length (``sk-`` 20,
# ``Bearer`` 20, Slack 10, AWS 16). Only ``sk-ant-``, ``ghp_`` and ``github_pat_``
# match as bare prefixes. Prose that merely names a key does not match.
#
# Known limits: a short password or one with spaces is not found (``password=abc``),
# and neither is a secret that matches none of these forms.
REPORT_SECRET_PATTERNS = (
    ("AWS access key", re.compile(r"A(?:KIA|SIA)[0-9A-Z]{16}")),  # long-term, STS
    ("Anthropic key", re.compile(r"(?<![A-Za-z0-9])sk-ant-")),
    # Not inside a word (risk-adjusted-..., disk-backed-...); also sk-proj-, sk-or-v1-.
    ("API key", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"ghp_|github_pat_")),
    ("Slack token", re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}")),
    # Bounded, not ``.*``: that backtracks quadratically on a line of BEGIN markers,
    # and ``re`` holds the GIL, so a slow scan would freeze every thread.
    ("private key", re.compile(r"-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----")),
    ("bearer token", re.compile(r"Bearer [A-Za-z0-9._-]{20,}")),
    (
        "credential assignment",
        # The log is JSON lines: a key may be followed by its closing quote, and a
        # quote inside a logged string is escaped, so both are allowed around it.
        # The name may end in _key or _access_key (secret_key, AWS_SECRET_ACCESS_KEY),
        # be plural (tokens) or hyphenated (x-api-key), or just contain these
        # (access_token); other suffixes (password_column, token_count) are not it.
        re.compile(
            r"(api[_-]?key|secret(?:[_-]access[_-]key|[_-]key)?|tokens?|password"
            r"|passwd|LAMIN_API_KEY)"
            r"\\?[\"']?\s*[=:]\s*\\?[\"']?[^\s\"'\\]{12,}",
            re.IGNORECASE,
        ),
    ),
)
# Decision (Nov 2026): for a named instance the environment record leaves the workspace (package
# names only). True: a named instance gets it, its file kept in the instance's storage.
# False: it gets none; the workspace catalog is not affected.
INSTANCE_ENVIRONMENT = True
# The ``_hash_type`` of a record whose ``hash`` is made from its SHA-256, because
# another version at the same key has LaminDB's hash for the same first and last chunk.
OWN_HASH_TYPE = "sha256"
# A named instance copies a file at most this often while it keeps changing.
COPY_ATTEMPTS = 2
RUN_REFERENCE_TYPES = {"session": "lobster_session", "command": "lobster_command"}
_STATUS_CODES = {"completed": 0, "errored": 1, "aborted": 2}


@dataclass(frozen=True)
class ReplayRun:
    """A replay's Run; ``session_uid`` is the Run that exported the notebook, if known."""

    uid: str
    session_uid: Optional[str]


class CatalogBusy(RuntimeError):
    """Another process holds the workspace catalog lock."""


class CatalogReadOnly(RuntimeError):
    """The catalog cannot be written to."""


class CatalogUnavailable(RuntimeError):
    """The engine cannot run in this process; the message says why."""


def workspace_settings_dir(workspace: Path) -> Path:
    return catalog_layout(workspace).catalog_dir / SETTINGS_SUBDIR


def workspace_settings_conflict(workspace: Path) -> Optional[str]:
    """Why the workspace catalog cannot start in this process, or None.

    Lamin reads its settings directory once, when ``lamindb_setup`` is first
    imported. If that happened before with another directory, connecting here would
    write into the user's own Lamin settings.
    """
    if "lamindb_setup" not in sys.modules:
        return None
    wanted = str(workspace_settings_dir(workspace))
    if os.environ.get("LAMIN_SETTINGS_DIR") == wanted:
        return None
    return "LaminDB was already loaded in this process with other settings"


class LaminCatalog:
    """``Catalog`` engine backed by LaminDB; synchronous, may raise."""

    def __init__(
        self,
        workspace_path: Path,
        instance: Optional[str] = None,
        *,
        create: bool = True,
    ):
        self.layout = catalog_layout(workspace_path, instance=bool(instance))
        self.workspace = self.layout.workspace
        self.home = self.layout.home  # the lock, the hidden records, the instance name
        self._create = create
        self.mode = "instance" if instance else "workspace"
        if self.mode == "workspace" and self.layout.runs_refusal:
            raise CatalogUnavailable(self.layout.runs_refusal)
        self._run: Any = None
        self._replay_session: Any = None
        self._transform: Any = None
        # (Lobster version, environment text, its hash): read once per version, so
        # a switch followed by exit does not scan the installed packages twice.
        self._environment_cache: Optional[tuple] = None
        self._warned: set = set()
        # keep_path's answers, by save path: (record uid, where it is kept).
        self._pending: Dict[str, tuple] = {}
        if self.mode == "workspace":
            self.slug = self._open_workspace_catalog()
        else:
            self.slug = self._connect_instance(instance)
        import lamindb as ln  # after connect: lamindb binds to the connected instance

        self.ln = ln
        self._storage_root = Path(str(self._setup.settings.instance.storage.root))

    # -- connection --------------------------------------------------------------

    def _open_workspace_catalog(self) -> str:
        conflict = workspace_settings_conflict(self.workspace)
        if conflict:
            raise CatalogUnavailable(conflict)
        catalog_dir = self.layout.catalog_dir
        settings_dir = catalog_dir / SETTINGS_SUBDIR
        settings_dir.mkdir(parents=True, exist_ok=True)
        os.environ["LAMIN_SETTINGS_DIR"] = str(settings_dir)
        # LaminDB creates its cache directory on first use (by default in the
        # user's cache folder); keep it in the catalog folder unless the user chose one.
        os.environ.setdefault("LAMIN_CACHE_DIR", str(catalog_dir / CACHE_SUBDIR))
        # The workspace catalog never logs in: with LAMIN_API_KEY set, LaminDB would
        # log in to Lamin's servers on first use. Its user is read once, without it.
        api_key = os.environ.pop("LAMIN_API_KEY", None)
        try:
            import lamindb_setup

            _log_to_stderr(lamindb_setup)
            lamindb_setup.settings.user  # noqa: B018 - loaded and cached here
        finally:
            if api_key is not None:
                os.environ["LAMIN_API_KEY"] = api_key

        self._setup = lamindb_setup
        config_path = self.layout.config_file
        config = _read_json(config_path)
        slug = config.get("slug") if isinstance(config.get("slug"), str) else None
        if slug:
            from lamindb_setup.core._settings_store import instance_settings_file

            owner, _, name = slug.partition("/")
            if not instance_settings_file(name, owner).is_file():
                # Connecting now would ask Lamin's servers for a local instance, and
                # init would refuse a storage location that is already managed.
                raise CatalogUnavailable(
                    f"its settings are missing from {CATALOG_DIR}/{SETTINGS_SUBDIR}; "
                    f"restore that folder, or move {CATALOG_DIR} aside to start a "
                    "new catalog"
                )
            lamindb_setup.connect(slug, _write_settings=False)
            return slug
        if not self._create:
            raise CatalogUnavailable("this workspace has no catalog yet")
        lamindb_setup.init(
            storage=str(self.layout.root),
            name=instance_name(self.home),
            _write_settings=False,
        )
        isettings = lamindb_setup.settings.instance
        # Keep only the instance's own settings file, so a later process can connect.
        isettings._persist(write_to_disk=True, write_current_instance_file=False)
        slug = isettings.slug
        _write_json(config_path, {"slug": slug, "created_by": "lobster"})
        return slug

    def _connect_instance(self, instance: str) -> str:
        import lamindb_setup

        _log_to_stderr(lamindb_setup)
        self._setup = lamindb_setup
        owner, _, name = instance.rpartition("/")
        root = _stored_storage_root(owner, name) if owner and name else None
        if root is not None and _is_remote_root(root):
            # Refused before connect, which would reach the remote storage first.
            raise CatalogUnavailable(_remote_storage_reason(instance, root))
        try:
            lamindb_setup.connect(instance, _write_settings=False)
        except Exception as exc:
            from lamindb_setup.core._settings_store import instance_settings_file

            if owner and name and not instance_settings_file(name, owner).is_file():
                # LaminDB then needs (anonymous: cannot use) Lamin's servers.
                raise CatalogUnavailable(
                    f"the LaminDB instance {instance} is not set up on this machine "
                    f"({type(exc).__name__})"
                ) from exc
            raise
        isettings = lamindb_setup.settings.instance
        if isettings.storage.type != "local":
            raise CatalogUnavailable(
                _remote_storage_reason(isettings.slug, isettings.storage.root)
            )
        return isettings.slug

    # -- Catalog -----------------------------------------------------------------

    def begin_run(self, session: SessionInfo) -> None:
        if self._run is not None:
            self.end_run("completed")
        self._run = self._new_run(session).save()

    def _new_run(self, session: SessionInfo):
        """The Run a session or command is recorded as, not yet saved.

        Decision (Nov 2026): one Run per session or command, Lobster's step log as its
        report. Per-tool child Runs (``initiated_by_run``) would replace this helper
        and extend the port: a tool call has no event in it yet.
        """
        run = self.ln.Run(
            self._pipeline(session.lobster_version),
            name=session.session_id,
            reference=session.session_id,
            reference_type=RUN_REFERENCE_TYPES.get(session.kind, "lobster_session"),
            status="started",
        )
        run.extra_data = {
            "lobster_version": session.lobster_version,
            "session_id": session.session_id,
            "kind": session.kind,
        }
        return run

    def keep_path(self, path: Path) -> Optional[Path]:
        """Where the recorded, unchanged version at ``path`` is to be kept, or None.

        Changes no file: the caller keeps the bytes there right before writing,
        and ``register_artifact`` points the record there once it finds them. When the file cannot be
        read, the save overwrites it and the record says so.
        """
        path = Path(path)
        self._pending.pop(str(path.resolve()), None)
        if self.mode != "workspace" or not path.is_file():
            return None
        latest = self._record_at(path)
        if latest is None:
            return None
        current = self._hash_or_none(path)
        if current is None:
            self._save_tag(latest, "unreadable", f"{self._storage_key(path)} at write")
            return None
        # The record whose bytes the file holds, not only the newest at that name.
        record = next(
            (r for r in self._records_at(path) if _holds_sha(r, path, current)), None
        )
        if record is None:
            return None  # already changed: register_artifact marks it superseded
        target = self._version_file(record, path)
        self._pending[str(path.resolve())] = (record.uid, target)
        return target

    def register_artifact(
        self,
        path: Path,
        *,
        key: str,
        kind: str,
        otype: Optional[str],
        description: str,
        meta: Dict[str, Any],
    ) -> Optional[ArtifactRef]:
        if self.mode == "instance":
            return self._register_copy(
                Path(path),
                key=key,
                kind=kind,
                otype=otype,
                description=description,
                meta=meta,
            )
        ln = self.ln
        not_kept = self._adopt_kept(Path(path))
        back = False
        sha256 = self._hash_or_none(Path(path))

        def build(skip_hash_lookup: bool):
            # description=None: LaminDB would otherwise overwrite the description of
            # an existing record when identical content is saved again.
            return ln.Artifact(
                Path(path),
                key=key,
                kind=kind,
                otype=otype,
                description=None,
                run=self._run if self._run is not None else False,
                skip_hash_lookup=skip_hash_lookup,
            )

        artifact = build(False)
        if not artifact._state.adding and artifact.kind == RUN_ARTIFACT_KIND:
            # LaminDB matched what it records about a run: this file is a record
            # of its own, not that one.
            artifact = build(True)
        if (
            sha256 is not None
            and not artifact._state.adding
            and not _holds_sha(artifact, Path(path), sha256)
        ):
            # LaminDB's hash covers only the first and last chunk of a large file:
            # this record has the same hash and other bytes.
            artifact = self._record_of_bytes(key, sha256) or build(True)
            if artifact._state.adding:
                # (storage, key, hash) is unique and the other version holds LaminDB's.
                artifact.hash = _own_hash(sha256)
                artifact._hash_type = OWN_HASH_TYPE
        existing = not artifact._state.adding
        if existing and self._restore(artifact, Path(path)):
            # The same bytes as a version kept aside before this save: the record
            # comes back to the file name and the moved copy is dropped.
            pass
        elif existing and artifact._storage_ongoing:
            # An earlier save of this record never completed, so its own file is
            # not trusted: ``path`` (whose bytes LaminDB just hashed) becomes its file.
            if self.mode == "workspace":
                if self._stored_path(artifact).resolve() != Path(path).resolve():
                    self._relocate(artifact, Path(path))
                artifact._storage_ongoing = False
            existing = False
        elif existing and not self._recorded_file_intact(artifact, Path(path)):
            # The recorded file no longer holds these bytes (it was overwritten
            # while the write guard was not available, or cannot be read), so
            # ``path`` is the copy to keep: point the record at it.
            self._relocate(artifact, Path(path))
            existing = False
        elif not existing:
            artifact.description = description or None
            artifact.extra_data = {"lobster": _with_sha256(meta, sha256)}
        elif self._unmark_superseded(artifact, Path(path)):
            # Back at its name with its own bytes: the latest again, and records
            # saved there in between no longer hold theirs.
            self._make_latest(artifact)
            back = True
        artifact = artifact.save()
        if not_kept is not None and not_kept.hash != artifact.hash:
            self._save_tag(not_kept, "not_kept", "overwritten before it could be kept")
            self._warn_once(
                "A recorded version of %s was overwritten before it could be kept; "
                "its record is marked.",
                self._storage_key(Path(path)),
            )
        if not existing or back:
            self._mark_superseded(artifact, Path(path))
        # The file the record points at now (``artifact.path`` may still name the
        # local file LaminDB was given).
        stored = self._stored_path(artifact) if self.mode == "workspace" else None
        return ArtifactRef(
            uid=artifact.uid,
            key=artifact.key,
            path=stored if stored is not None else Path(str(artifact.path)),
            version=artifact.version,
            existing=existing,
        )

    def _register_copy(
        self,
        path: Path,
        *,
        key: str,
        kind: str,
        otype: Optional[str],
        description: str,
        meta: Dict[str, Any],
    ) -> Optional[ArtifactRef]:
        """Instance mode: copy the bytes into the instance's storage, register there."""
        ln = self.ln
        folder = self._storage_root / "lobster" / instance_name(self.home) / key
        staged = None
        for _ in range(COPY_ATTEMPTS):
            current = _sha256(path)
            records = self._records_of_bytes(path, current)
            record = next((r for r in records if Path(str(r.path)).is_file()), None)
            if record is not None:
                break
            # The bytes that are stored give the record and the copy's name: a file
            # rewritten since it was read is copied again, with the hash of the copy.
            staged = _staged_copy(path, folder, current)
            if staged is not None:
                break
        else:
            self._warn_once(
                "%s changed while it was being copied; it is not recorded.", key
            )
            return None
        if record is None and records:
            # Recorded, but its copy left the instance's storage: put the bytes back.
            stored = Path(str(records[0].path))
            if self._in_storage(stored):
                stored.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, stored)
                record = records[0]
        if record is None:
            copy = folder / (current + path.suffix)
            os.replace(staged, copy)

            def build(skip_hash_lookup: bool):
                return ln.Artifact(
                    copy,
                    key=key,
                    kind=kind,
                    otype=otype,
                    description=description or None,
                    run=self._run if self._run is not None else False,
                    skip_hash_lookup=skip_hash_lookup,
                )

            artifact = build(False)
            if not artifact._state.adding:
                # LaminDB matched a record at this key with the same hash and other
                # bytes (its hash covers the first and last chunk of a large file):
                # this is a record of its own, with a hash made from the SHA-256.
                artifact = build(True)
                artifact.hash = _own_hash(current)
                artifact._hash_type = OWN_HASH_TYPE
            artifact.extra_data = {"lobster": _with_sha256(meta, current)}
            artifact = artifact.save()
            existing = False
        else:
            # The same bytes again: as LaminDB does for a hash match, this run
            # recreated the record (unless it read it as an input).
            artifact, existing = record, True
            run = self._run
            if run is not None and artifact.run_id != run.id:
                if not artifact.input_of_runs.filter(id=run.id).exists():
                    artifact.recreating_runs.add(run)
        return ArtifactRef(
            uid=artifact.uid,
            key=artifact.key,
            path=Path(str(artifact.path)),
            version=artifact.version,
            existing=existing,
        )

    def note_unrecorded(self, path: Path) -> None:
        path = Path(path)
        if self.mode != "workspace" or not path.is_file():
            return
        current = self._hash_or_none(path)
        if current is None:
            return
        stale = [
            record
            for record in self._records_at(path)
            if not _holds_sha(record, path, current)
            and "superseded_by" not in ((record.extra_data or {}).get("lobster") or {})
        ]
        for record in stale:
            self._save_tag(record, "superseded_by", "unrecorded")
        if stale:
            self._warn_once(
                "%s was saved while the catalog was busy and is not recorded; "
                "the earlier record of it is marked as superseded.",
                self._storage_key(path),
            )

    def link_input(self, path: Path) -> Optional[ArtifactRef]:
        if self._run is None:
            return None
        ln = self.ln
        record = next(
            iter(self._records_of_bytes(Path(path), _sha256(Path(path)))), None
        )
        if record is None:
            return None
        artifact = ln.Artifact.get(record.uid, is_run_input=self._run)
        return ArtifactRef(
            uid=artifact.uid,
            key=artifact.key,
            path=Path(str(artifact.path)),
            version=artifact.version,
            existing=True,
        )

    def register_transform(
        self, *, key: str, kind: str, source: str, description: str
    ) -> Optional[TransformRef]:
        # Decision 11 Nov 2026: the notebook source is scanned like the step log;
        # a hit registers nothing and the notebook file stays as it is.
        reason = self._scan_for_secret(source.encode("utf-8", "replace"), "notebook")
        if reason is not None:
            self._warn_once("Notebook not recorded: %s.", reason)
            self._note_on_run("unregistered_secret_scan", key)
            return None
        ln = self.ln
        transform = ln.Transform(
            key=key, kind=kind, source_code=source, description=description
        ).save()
        if self._run is not None and transform.run_id is None:
            transform.run = self._run
            transform.save()
        return TransformRef(uid=transform.uid, key=transform.key)

    def end_run(
        self,
        status: RunStatus,
        report_path: Optional[Path] = None,
        environment_path: Optional[Path] = None,
    ) -> None:
        run, self._run = self._run, None
        if run is None:
            return
        run.finished_at = datetime.now(timezone.utc)
        # No public setter yet; the Run model documents this field for completion.
        run._status_code = _STATUS_CODES.get(status, 1)
        run.save()
        # The run is closed first: a capture that outlasts the exit budget costs the
        # run its environment and report, not its finish.
        environment = self._environment(
            (run.extra_data or {}).get("lobster_version", "")
        )
        report = self._report(run, report_path)
        if environment is not None:
            run.environment = environment
        if report is not None:
            run.report = report
        if environment is not None or report is not None:
            run.save()

    # -- replays (``lobster replay``; not part of the port) -----------------------

    def begin_replay(
        self,
        *,
        key: str,
        source: str,
        params: Dict[str, Any],
        lobster_version: str,
    ) -> Optional[ReplayRun]:
        """Open the Run that records a replay of the notebook ``source`` (at ``key``).

        The Run is of the notebook's Transform, found by its source (registered now
        when the catalog has not seen these bytes) and initiated by the Run that
        exported it. It is not a session: it is not counted by ``overview()``, shown
        by ``list_runs()`` or made the current run of LaminDB. None when the notebook
        cannot be recorded (the secret scan) or a session's Run is open in this
        engine; nothing is created, and nothing is closed, then.
        """
        if self._run is not None:
            return None  # a session's Run is open: it is not ours to close
        ln = self.ln
        # One Transform per source: identical bytes under another key are the same one.
        transform = ln.Transform.filter(kind="notebook", source_code=source).first()
        if transform is None:
            ref = self.register_transform(
                key=key,
                kind="notebook",
                source=source,
                description="Notebook replayed by Lobster",
            )
            if ref is None:
                return None
            transform = ln.Transform.get(uid=ref.uid)
        # Only the key the session exported it under ties the replay to that
        # session: a copy under another key may come from another session.
        session = transform.run if transform.key == key else None
        fields: Dict[str, Any] = {"params": params, "name": f"replay:{Path(key).stem}"}
        if session is not None:
            fields["initiated_by_run"] = session
        run = ln.Run(transform=transform, **fields)
        run.extra_data = {
            "lobster_version": lobster_version,
            "kind": "replay",
            "notebook": key,
        }
        self._run = run.save()
        self._replay_session = session
        return ReplayRun(
            uid=run.uid, session_uid=None if session is None else session.uid
        )

    def replay_reference(self, stem: str) -> Optional[Path]:
        """The file the session saved as ``<stem>.h5ad``, if it still holds its bytes."""
        return self.replay_reference_state(stem)[0]

    def _session_runs(self, session: Any) -> List[Any]:
        """The Runs of one session: the one named, and any of the same name."""
        import lamindb as ln

        runs = [session]
        if getattr(session, "name", None):
            runs += list(
                ln.Run.filter(name=session.name, transform__kind=session.transform.kind)
                .exclude(id=session.id)
                .all()
            )
        return runs

    def replay_reference_state(self, stem: str) -> Tuple[Optional[Path], str]:
        """``(file, "ok")``, ``(None, "changed")`` or ``(None, "none")`` for ``<stem>.h5ad``.

        The reference is the session's record whose bytes ARE the workspace file
        (``data/<stem>.h5ad``, else ``<stem>.h5ad``). Never the newest record: a
        session that saved X, then Y, then X again holds X. A workspace file that
        no record of the session has is "changed", unless the session saved one set
        of bytes only and a later save the catalog knows replaced the file. Records
        that disagree with no file to settle it are "changed"; no record is "none".
        """
        from django.db.models import Q

        session = self._replay_session
        if session is None:
            return None, "none"
        # A session restored in another process has a Run of its own with the same
        # name; the files an earlier process saved belong to the session's Runs.
        sessions = list(self._session_runs(session))
        records = {}  # sha256 -> (record, its stored path)
        for path in (
            self.workspace / "data" / f"{stem}.h5ad",
            self.workspace / f"{stem}.h5ad",
        ):
            try:  # a project's catalog keys a file from the project folder
                key = path.resolve().relative_to(self.layout.root.resolve()).as_posix()
            except ValueError:
                continue
            found = (
                self._files()
                .filter(key=key)
                .filter(Q(run__in=sessions) | Q(recreating_runs__in=sessions))
                .distinct()
                .order_by("-created_at")
            )
            for record in found:
                path = self._stored_path(record)
                intact = self._holds(record, path)
                sha = _recorded_sha256(record) or (_sha256(path) if intact else None)
                if sha is not None and sha not in records:
                    records[sha] = (record, path, intact)
        if not records:
            return None, "none"
        files = [
            f
            for f in (
                self.workspace / "data" / f"{stem}.h5ad",
                self.workspace / f"{stem}.h5ad",
            )
            if f.is_file()
        ]
        for workspace_file in files:
            hit = records.get(_sha256(workspace_file))
            if hit is not None:
                return (hit[1] if hit[2] else workspace_file), "ok"
        if len(records) == 1:  # the session saved one set of bytes only
            ((_record, path, intact),) = records.values()
            # The file is gone, or a later save the catalog knows replaced it; a file
            # nobody recorded is an edit, and that is "changed".
            known = all(self._records_of_bytes(f, _sha256(f)) for f in files)
            if intact and known:
                return path, "ok"
        return None, "changed"

    def end_replay(
        self,
        status: RunStatus,
        *,
        verdict: str,
        report_path: Optional[Path],
        outputs: Sequence[Path],
    ) -> None:
        """Record the replay's outputs and verdict, attach its report, close the Run.

        An output whose bytes are already recorded (the replay recreated them) is
        linked to that record; no record is changed or moved. Any other output is a
        record of its own at its path in the workspace (``replays/<run uid>/...``),
        never a version of a saved file.
        """
        run = self._run
        if run is None:
            return
        try:
            for path in outputs:
                self._record_replay_output(Path(path))
        finally:
            self._run = self._replay_session = None
            run.finished_at = datetime.now(timezone.utc)
            run._status_code = _STATUS_CODES.get(status, 1)
            run.extra_data = {**(run.extra_data or {}), "verdict": verdict}
            run.save()
            report = self._report(run, report_path)
            if report is not None:
                run.report = report
                run.save()

    def _record_replay_output(self, path: Path) -> None:
        run = self._run
        try:
            key = path.resolve().relative_to(self.layout.root).as_posix()
            recorded = self._records_of_bytes(path, _sha256(path))
            if recorded:
                recorded[0].recreating_runs.add(run)
                return
            self.register_artifact(
                path,
                key=key,
                kind="dataset",
                otype=None,
                description="Output of a replay",
                meta={"replay": run.uid},
            )
        except Exception:
            logger.debug("Catalog: could not record a replay output", exc_info=True)
            self._warn_once("An output of a replay could not be recorded.")

    def status(self) -> CatalogStatus:
        isettings = self._setup.settings.instance
        return CatalogStatus(
            engine="lamindb",
            mode=self.mode,  # type: ignore[arg-type]
            instance=self.slug,
            location=str(self._storage_root),
            run=None if self._run is None else self._run.uid,
            details={
                "db": str(isettings.db),
                "lamindb": _version("lamindb-core"),
                "lamindb_setup": _version("lamindb-setup"),
            },
        )

    # -- browsing (``lobster lamin``; not part of the port) -----------------------

    def overview(self) -> Dict[str, Any]:
        ln = self.ln
        runs = ln.Run.filter(transform__key=PIPELINE_KEY)
        last = runs.order_by("-started_at").first()
        return {
            "artifacts": self._files().count(),
            "runs": runs.count(),
            "notebooks": ln.Transform.filter(kind="notebook").count(),
            "last_run": None if last is None else _run_row(last),
        }

    def list_artifacts(self, limit: int = 50) -> list:
        records = self._files().select_related("run").order_by("-created_at")[:limit]
        return [
            {
                "key": record.key,
                "version": record.version,
                "latest": record.is_latest,
                "uid": record.uid,
                "size": record.size,
                "created_at": _iso(record.created_at),
                "file": self._storage_key(self._stored_path(record)),
                "run": None if record.run is None else record.run.name,
                "notes": sorted(
                    k
                    for k in ((record.extra_data or {}).get("lobster") or {})
                    if k
                    in (
                        "superseded_by",
                        "relocated_from",
                        "unreadable",
                        "not_kept",
                        "orphan_copy",
                    )
                ),
            }
            for record in records
        ]

    def kept_versions(self) -> list:
        """The earlier versions whose file is a kept copy in the version store."""
        return [
            {
                "key": record.key,
                "version": record.version,
                "uid": record.uid,
                "file": self._storage_key(stored),
                "bytes": size,
            }
            for record, stored, size in self._kept_copies()
        ]

    def prune_versions(self, shown: list) -> Dict[str, list]:
        """Remove the kept copies ``shown`` (rows of ``kept_versions``); records stay.

        Only a copy that is still on offer, as shown, goes; anything else is
        skipped and listed. A record is marked as not kept before its file is
        removed, so a file is never gone without its record saying so. The whole
        prune holds the catalog lock, and a catalog that cannot be written to
        is refused before anything is removed.
        """
        from lobster.core.catalog.factory import LOCK_FILE
        from lobster.core.catalog.thread import _LockBusy, workspace_lock

        try:
            with workspace_lock(self.home / LOCK_FILE, wait=0):
                return self._prune(shown)
        except _LockBusy as exc:
            raise CatalogBusy(str(exc)) from exc

    def _prune(self, shown: list) -> Dict[str, list]:
        removed, failed, skipped = [], [], []
        offered = {
            (record.uid, self._storage_key(stored)): (record, stored, size)
            for record, stored, size in self._kept_copies()
        }
        wanted = list(dict.fromkeys((row["uid"], row["file"]) for row in shown))
        if any(key in offered for key in wanted):
            try:
                self._write_probe(offered[next(k for k in wanted if k in offered)][0])
            except Exception as exc:
                raise CatalogReadOnly(f"{type(exc).__name__}: {exc}") from exc
        versions = self._versions_dir()
        for key in wanted:
            if key not in offered:
                skipped.append({"file": key[1]})
                continue
            record, stored, size = offered[key]
            rel = key[1]
            try:  # marked first: a file never goes without its record saying so
                self._tag_strict(record, "not_kept", "removed by lobster lamin prune")
            except Exception:
                logger.debug("Catalog: could not mark %s", rel, exc_info=True)
                failed.append({"file": rel, "error": "NotMarked"})
                continue
            try:
                stored.unlink()
            except FileNotFoundError:
                pass  # already gone: removed, and the mark stays
            except OSError as exc:
                failed.append({"file": rel, "error": type(exc).__name__})
                try:
                    self._tag_strict(record, "not_kept", None, remove=True)
                except Exception:
                    logger.debug("Catalog: could not unmark %s", rel, exc_info=True)
                continue
            removed.append(
                {
                    "key": record.key,
                    "version": record.version,
                    "uid": record.uid,
                    "file": rel,
                    "bytes": size,
                }
            )
            for folder in stored.parents:  # the folders it leaves empty
                if folder == versions:
                    break
                try:
                    folder.rmdir()
                except OSError:
                    break
        return {"removed": removed, "failed": failed, "skipped": skipped}

    @staticmethod
    def _write_probe(record: Any) -> None:
        record.save()  # an update that changes nothing, but must be allowed

    def _kept_copies(self) -> list:
        """``(record, file, size)`` of each earlier version kept in the version store.

        What ``status`` counts, ``prune`` offers and removes: a record that is not
        the latest of its key, with a file that resolves to inside the store.
        """
        from django.db.models import Q

        prefix = f"{CATALOG_DIR}/{VERSIONS_SUBDIR}/"
        copies = []
        # A Q object: a keyword starting with "_" would be read as a feature name.
        for record in (
            self._files()
            .filter(Q(_real_key__startswith=prefix), is_latest=False)
            .order_by("created_at")
        ):
            stored = self._stored_path(record)
            try:
                if stored.is_file() and self._in_versions(stored):
                    copies.append((record, stored, stored.stat().st_size))
            except OSError:
                continue
        return copies

    def list_runs(self, limit: int = 50) -> list:
        from django.db.models import Q

        records = (
            self.ln.Run.filter(
                Q(transform__key=PIPELINE_KEY) | Q(extra_data__kind="replay")
            )
            .select_related("transform")
            .order_by("-started_at")[:limit]
        )
        return [_run_row(record) for record in records]

    # -- mapping helpers ---------------------------------------------------------

    def _pipeline(self, lobster_version: str):
        ln = self.ln
        if self._transform is not None and self._transform.version == lobster_version:
            return self._transform
        transform = ln.Transform.filter(
            key=PIPELINE_KEY, version=lobster_version
        ).first()
        if transform is None:
            transform = ln.Transform(
                key=PIPELINE_KEY,
                version=lobster_version,
                kind="pipeline",
                description="Lobster AI session",
                reference=PIPELINE_REFERENCE,
            ).save()
        self._transform = transform
        return transform

    def _files(self):
        """The recorded files a user sees (not what LaminDB records about a run)."""
        return self.ln.Artifact.filter().exclude(kind=RUN_ARTIFACT_KIND)

    def _environment(self, lobster_version: str):
        """The environment as a hidden run Artifact, or None.

        Read when the run ends, so the first saves of a session never wait for it.
        The same text is the same record, found by its hash. The file is named by
        that hash and kept where saved files are, not under ``.lamindb/``, where
        LaminDB refuses a path: in the workspace's own ``.lobster/`` folder, or, for a
        named instance, in the instance's storage. A failure costs the run its
        environment only.
        """
        if self.mode == "instance" and not INSTANCE_ENVIRONMENT:
            return None
        try:
            if (
                self._environment_cache is None
                or self._environment_cache[0] != lobster_version
            ):
                from lamindb_setup.core.hashing import hash_small_bytes

                text = environment_text(lobster_version).encode()
                self._environment_cache = (
                    lobster_version,
                    text,
                    hash_small_bytes(text),
                )
            _, data, digest = self._environment_cache
            record = self.ln.Artifact.filter(
                hash=digest, kind=RUN_ARTIFACT_KIND
            ).first()
            if record is not None:
                # Its own file, which another workspace on the same instance may
                # have written: no second copy without a record.
                path = Path(str(record.path))
                if not self._in_storage(path):
                    raise ValueError(
                        f"the record's file is outside the storage: {path}"
                    )
            else:
                path = self._record_dir(ENVIRONMENTS_SUBDIR) / f"{digest}.txt"
            if not path.is_file() or path.read_bytes() != data:
                _write_whole(path, data)  # also puts back a record's missing file
            if record is None:
                record = self.ln.Artifact(
                    path,
                    description="environment",
                    kind=RUN_ARTIFACT_KIND,
                    run=False,
                ).save()
            return record
        except Exception:
            logger.debug("Catalog: could not record the environment", exc_info=True)
            self._warn_once("The environment of the run could not be recorded.")
            return None

    def _report(self, run: Any, report_path: Optional[Path]):
        """A snapshot of the run's step log as a hidden run Artifact, or None.

        The live log keeps growing after the run ends, so the record points at a
        copy made now, up to the last complete line. It is kept in the workspace's
        ``.lobster/reports/`` folder (LaminDB refuses a path under ``.lamindb/``) or,
        for a named instance, in the instance's storage. The bytes are scanned
        first: nothing is written for a log that looks like it holds a secret.
        Nothing is recorded either when the log is missing, too large
        (``REPORT_MAX_BYTES``) or cannot be read or copied; the run still ends.

        Decision 11 Nov 2026: a named instance gets the step log copied, subject to the scan.
        """
        if report_path is None:
            return None
        try:
            source = Path(report_path)
            size = source.stat().st_size
            if size > REPORT_MAX_BYTES:
                self._warn_once(
                    "The step log of a run is not recorded: it is larger than %d bytes.",
                    REPORT_MAX_BYTES,
                )
                return None
            with source.open("rb") as handle:
                data = handle.read(size)
            # Up to the last complete line: a tool thread may still be writing.
            data = data[: data.rfind(b"\n") + 1]
            if not data:
                return None
            reason = self._scan_for_secret(data, "step log")
            if reason is not None:
                self._warn_once("Report not recorded: %s.", reason)
                return None
            suffix = ".json" if source.suffix == ".json" else ".jsonl"
            snapshot = self._record_dir(REPORTS_SUBDIR) / f"{run.uid}{suffix}"
            _write_whole(snapshot, data)
            try:
                return self.ln.Artifact(
                    snapshot,
                    description=f"Report of run {run.uid}",
                    kind=RUN_ARTIFACT_KIND,
                    run=False,
                ).save()
            except Exception:
                snapshot.unlink(missing_ok=True)  # no copy without a record
                raise
        except Exception:
            logger.debug("Catalog: could not record the step log", exc_info=True)
            self._warn_once("The step log of a run could not be recorded.")
            return None

    def _scan_for_secret(self, data: bytes, what: str) -> Optional[str]:
        """Why ``data`` (the ``what`` of a run) is kept out of the catalog, or None.

        Decision 11 Nov 2026: secrets scan on, fail closed (an error while scanning
        records nothing). Runs on the bytes before any copy or record exists.
        """
        text = data.decode("utf-8", errors="replace")
        for label, pattern in REPORT_SECRET_PATTERNS:
            if pattern.search(text):
                return f"the {what} looks like it contains a secret ({label})"
        return None

    def _record_dir(self, subdir: str) -> Path:
        """Where the files of the hidden run records of this catalog are kept.

        In the workspace's own ``.lobster/`` folder, or, for a named instance, in
        that instance's storage next to the files it holds for this workspace.
        """
        if self.mode == "workspace":
            return self.home / ".lobster" / subdir
        return self._storage_root / "lobster" / instance_name(self.home) / subdir

    def _storage_key(self, path: Path) -> Optional[str]:
        try:
            return path.resolve().relative_to(self._storage_root.resolve()).as_posix()
        except ValueError:
            return None

    def _in_storage(self, path: Path) -> bool:
        """Whether ``path`` is a local file path inside this instance's storage."""
        return path.is_absolute() and path.resolve().is_relative_to(
            self._storage_root.resolve()
        )

    def _records_at(self, path: Path):
        """Artifacts whose file is ``path`` in this instance's storage, newest first."""
        rel = self._storage_key(path)
        if rel is None:
            return self.ln.Artifact.filter(id__in=[])
        from django.db.models import Q

        storage = self._setup.settings.instance.storage.record
        return (
            self.ln.Artifact.filter(storage=storage)
            .filter(
                Q(_real_key=rel)
                | Q(_real_key__isnull=True, key=rel, _key_is_virtual=False)
            )
            .order_by("-created_at")
        )

    def _records_of_bytes(self, path: Path, sha256: str) -> list:
        """Recorded files holding the bytes of ``path`` (``sha256``), newest first.

        LaminDB's hash only narrows the candidates: it covers the first and last
        chunk of a large file. A record this engine made with another hash for the
        same reason (``OWN_HASH_TYPE``) is found by its own.
        """
        from django.db.models import Q

        candidates = self._files().filter(
            Q(hash=_lamin_hash(path)) | Q(hash=_own_hash(sha256))
        )
        return [
            r for r in candidates.order_by("-created_at") if _holds_sha(r, path, sha256)
        ]

    def _record_of_bytes(self, key: str, sha256: str):
        """The record at ``key`` that holds the bytes with ``sha256``, if any."""
        return next(
            (r for r in self._files().filter(key=key) if _recorded_sha256(r) == sha256),
            None,
        )

    def _record_at(self, path: Path):
        """The artifact whose file is ``path`` in this instance's storage, if any."""
        return self._records_at(path).first()

    def _recorded_file_intact(self, artifact: Any, path: Path) -> bool:
        """Whether an existing record's own file still holds its recorded bytes."""
        if self.mode != "workspace":
            return True  # files in an instance's storage are managed by LaminDB
        recorded = self._stored_path(artifact)
        if recorded.resolve() == path.resolve():
            return True
        if not recorded.is_file():
            return False
        current = self._hash_or_none(recorded)
        if current is None:
            self._tag(
                artifact, "unreadable", f"{self._storage_key(recorded)} at re-save"
            )
            return False
        return _holds_sha(artifact, recorded, current)

    def _point_at(self, artifact: Any, path: Path) -> bool:
        """Point ``artifact`` at ``path`` in this instance's storage (not saved)."""
        rel = self._storage_key(path)
        if rel is None:
            return False
        artifact.storage = self._setup.settings.instance.storage.record
        artifact._real_key = None if rel == artifact.key else rel
        artifact._key_is_virtual = rel != artifact.key
        return True

    def _versions_dir(self) -> Path:
        return self._storage_root / CATALOG_DIR / VERSIONS_SUBDIR

    def _version_file(self, record: Any, path: Path) -> Path:
        rel = self._storage_key(path) or path.name
        version = str(record.version or "")
        folder = self._versions_dir() / rel
        name = (
            f"v{version}{path.suffix}"
            if re.fullmatch(r"[0-9A-Za-z._-]+", version)
            else f"{record.uid}{path.suffix}"
        )
        target = folder / name
        if target.exists():
            target = folder / f"{target.stem}-{record.uid}{path.suffix}"
        return target

    def _adopt_kept(self, path: Path) -> Any:
        """Point the record ``keep_path`` named for ``path`` at its kept copy.

        Returns that record when the kept copy is not there with its bytes (the
        save went ahead unprepared), else None.
        """
        pending = self._pending.pop(str(path.resolve()), None)
        if pending is None:
            return None
        uid, target = pending
        record = self.ln.Artifact.filter(uid=uid).first()
        if record is None:
            return None
        if not self._holds(record, target):
            return record
        self._move_record(record, target)
        logger.info(
            "Catalog: kept %s as %s",
            self._storage_key(path),
            self._storage_key(target),
        )
        return None

    def _holds(self, record: Any, path: Path) -> bool:
        """Whether ``path`` is a file with ``record``'s bytes (size, then hash)."""
        try:
            if not path.is_file() or path.stat().st_size != record.size:
                return False
            return _holds_sha(record, path, _sha256(path))
        except OSError:
            return False

    def _move_record(self, record: Any, path: Path) -> None:
        """Point ``record`` at ``path``, which holds its bytes, and save it."""
        self._point_at(record, path)
        record._storage_ongoing = False  # its bytes were just verified
        lobster = (record.extra_data or {}).get("lobster") or {}
        if "superseded_by" in lobster:
            extra = dict(record.extra_data)
            extra["lobster"] = {
                k: v for k, v in lobster.items() if k != "superseded_by"
            }
            record.extra_data = extra
        record.save()

    def _find_kept(self, record: Any, path: Path) -> Optional[Path]:
        """A copy of ``record``'s bytes kept for ``path`` in the version store."""
        rel = self._storage_key(path)
        folder = self._versions_dir() / rel if rel else None
        if folder is None or not folder.is_dir():
            return None
        for candidate in sorted(folder.iterdir()):
            if self._holds(record, candidate):
                return candidate
        return None

    def _in_versions(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self._versions_dir().resolve())
        except ValueError:
            return False
        return True

    def _restore(self, artifact: Any, path: Path) -> bool:
        """Re-point a record kept in the version store at ``path`` (same bytes)."""
        if self.mode != "workspace" or artifact._storage_ongoing:
            return False
        if self._storage_key(path) != artifact.key:
            return False  # same bytes under another name: leave both as they are
        stored = self._stored_path(artifact)
        if not self._in_versions(stored):
            return False
        if not self._point_at(artifact, path):
            return False
        self._make_latest(artifact)
        lobster = (artifact.extra_data or {}).get("lobster") or {}
        if "not_kept" in lobster:  # its copy was pruned; the file name holds it again
            extra = dict(artifact.extra_data)
            extra["lobster"] = {k: v for k, v in lobster.items() if k != "not_kept"}
            artifact.extra_data = extra
        artifact.save()
        try:
            stored.unlink(missing_ok=True)  # missing: pruned
        except OSError as exc:
            rel = self._storage_key(stored)
            self._save_tag(artifact, "orphan_copy", rel)
            self._warn_once(
                "%s could not be removed (%s); it is an extra copy of %s.",
                rel,
                type(exc).__name__,
                artifact.key,
            )
            return True
        try:
            stored.parent.rmdir()  # only when no other version is kept there
        except OSError:
            pass
        return True

    def _relocate(self, artifact: Any, path: Path) -> None:
        old = str(self._stored_path(artifact))
        if not self._point_at(artifact, path):
            return
        rel = self._storage_key(path)
        if rel == artifact.key:
            self._make_latest(artifact)
        extra = dict(artifact.extra_data or {})
        lobster = dict(extra.get("lobster") or {})
        lobster.pop("superseded_by", None)
        lobster["relocated_from"] = self._storage_key(Path(old)) or old
        extra["lobster"] = lobster
        artifact.extra_data = extra
        self._warn_once(
            "A recorded file was overwritten before the catalog could guard it; its "
            "record now points at %s, which holds the recorded bytes.",
            rel,
        )

    def _mark_superseded(self, artifact: Any, path: Path) -> None:
        """Tag older records at ``path`` whose bytes the file no longer holds."""
        if self.mode != "workspace":
            return
        stale = []
        for record in self._records_at(path).exclude(id=artifact.id):
            if record.hash == artifact.hash or "superseded_by" in (
                (record.extra_data or {}).get("lobster") or {}
            ):
                continue
            kept = self._find_kept(record, path)
            if kept is not None:  # moved by a save whose registration was lost
                self._move_record(record, kept)
                continue
            stale.append(record)
        for record in stale:
            extra = dict(record.extra_data or {})
            lobster = dict(extra.get("lobster") or {})
            lobster["superseded_by"] = artifact.uid
            extra["lobster"] = lobster
            record.extra_data = extra
            record.save()
        if stale:
            self._warn_once(
                "%s was overwritten; %d earlier recorded version(s) of it are no "
                "longer on disk and are marked as superseded in the catalog.",
                self._storage_key(path),
                len(stale),
            )

    def _unmark_superseded(self, artifact: Any, path: Path) -> bool:
        """The file holds this record's bytes again: drop its superseded mark.

        True when the mark was dropped (the record is back at its own file).
        """
        lobster = (artifact.extra_data or {}).get("lobster") or {}
        if "superseded_by" not in lobster:
            return False
        if self.mode != "workspace" or self._stored_path(artifact).resolve() != (
            path.resolve()
        ):
            return False
        extra = dict(artifact.extra_data)
        extra["lobster"] = {k: v for k, v in lobster.items() if k != "superseded_by"}
        artifact.extra_data = extra
        return True

    def _stored_path(self, artifact: Any) -> Path:
        """Where the record says its file is (``artifact.path`` may name the new file)."""
        from lamindb.core.storage.paths import auto_storage_key_from_artifact

        return Path(str(artifact.storage.root)) / auto_storage_key_from_artifact(
            artifact
        )

    def _hash_or_none(self, path: Path) -> Optional[str]:
        """The SHA-256 of ``path``, or None when it cannot be read (warned once)."""
        try:
            return _sha256(path)
        except OSError:
            self._warn_once(
                "%s could not be read to check it against the catalog; the save "
                "goes ahead and the record is marked as not checked.",
                self._storage_key(path) or str(path),
            )
            return None

    @staticmethod
    def _tag(record: Any, name: str, value: Any) -> None:
        """Set ``extra_data["lobster"][name]`` on ``record`` (saved with it)."""
        extra = dict(record.extra_data or {})
        lobster = dict(extra.get("lobster") or {})
        lobster[name] = value
        extra["lobster"] = lobster
        record.extra_data = extra

    def _make_latest(self, artifact: Any) -> None:
        """The record at the file name is the latest of its key; the others are not."""
        others = self.ln.Artifact.filter(key=artifact.key, is_latest=True).exclude(
            id=artifact.id
        )
        for other in others:
            other.is_latest = False
            other.save()
        artifact.is_latest = True

    def _tag_strict(
        self, record: Any, name: str, value: Any, remove: bool = False
    ) -> None:
        """Save a note on ``record`` (or take it off); a failure is raised."""
        # A newer version may have changed this flag since the record was read.
        record.refresh_from_db(fields=["is_latest"])
        self._tag(record, name, value)
        if remove:
            extra = dict(record.extra_data)
            extra["lobster"] = {k: v for k, v in extra["lobster"].items() if k != name}
            record.extra_data = extra
        record.save()

    def _save_tag(self, record: Any, name: str, value: Any) -> None:
        # A note that cannot be saved must not cost the save it describes.
        try:
            self._tag_strict(record, name, value)
        except Exception:
            logger.debug("Catalog: could not save tag %s", name, exc_info=True)
            self._warn_once("A note on a recorded file could not be saved.")

    def _note_on_run(self, name: str, key: str) -> None:
        """Add ``key`` to the list ``extra_data["lobster"][name]`` of the open run."""
        run = self._run
        if run is None:
            return
        try:
            noted = list(((run.extra_data or {}).get("lobster") or {}).get(name) or [])
            if key not in noted:
                noted.append(key)
            self._tag(run, name, noted)
            run.save()
        except Exception:
            logger.debug("Catalog: could not note %s on the run", name, exc_info=True)

    def _warn_once(self, message: str, *args: Any) -> None:
        if message in self._warned:
            return
        self._warned.add(message)
        logger.warning("Catalog: " + message, *args)


def _iso(value: Any) -> Optional[str]:
    return None if value is None else value.isoformat(timespec="seconds")


def _run_row(run: Any) -> Dict[str, Any]:
    extra = run.extra_data if isinstance(run.extra_data, dict) else {}
    replay = extra.get("kind") == "replay"
    return {
        "kind": "replay" if replay else "session",
        "verdict": extra.get("verdict") if replay else None,
        "name": run.name,
        "status": run.status,
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
        "lobster_version": run.transform.version,
        "outputs": run.output_artifacts.count(),
        "inputs": run.input_artifacts.count(),
        "uid": run.uid,
        "report": run.report_id is not None,
        "environment": run.environment_id is not None,
    }


def _remote_storage_reason(slug: str, root: Any) -> str:
    return (
        f"this release records only into local storage; the LaminDB "
        f"instance {slug} is remote ({root})"
    )


def _stored_storage_root(owner: str, name: str) -> Optional[str]:
    """The storage root in the user's settings file of an instance, if there is one."""
    from dotenv import dotenv_values
    from lamindb_setup.core._settings_store import instance_settings_file

    settings_file = instance_settings_file(name, owner)
    if not settings_file.is_file():
        return None
    return dotenv_values(settings_file).get("lamindb_instance_storage_root")


def _is_remote_root(root: str) -> bool:
    if root.strip().lower().startswith("create-s3"):  # LaminDB's managed-bucket form
        return True
    scheme, sep, _ = root.partition("://")
    return bool(sep) and scheme.lower() not in ("", "file")


def _copy_file(source: Path, target: Path) -> None:
    """Copy ``source`` to ``target`` whole or not at all."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.name}.", suffix=".partial", delete=False
    ) as handle:
        partial = Path(handle.name)
    try:
        shutil.copy2(source, partial)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def _staged_copy(source: Path, folder: Path, sha256: str) -> Optional[Path]:
    """A copy of ``source`` in ``folder`` whose bytes have ``sha256``, else None.

    The caller names and places the copy; one that holds other bytes (the source
    was rewritten after it was hashed) is removed.
    """
    staged = folder / f".staged-{secrets.token_hex(4)}.partial"
    try:
        _copy_file(source, staged)
        if _sha256(staged) == sha256:
            return staged
        staged.unlink()
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return None


def _write_whole(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` whole or not at all."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.name}.", suffix=".partial", delete=False
    ) as handle:
        partial = Path(handle.name)
    try:
        partial.write_bytes(data)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def _lamin_hash(path: Path) -> str:
    from lamindb_setup.core.hashing import hash_file

    return hash_file(Path(path))[1]


def _sha256(path: Path) -> str:
    """The SHA-256 of the whole file, read in blocks.

    LaminDB's own hash covers only the first and last chunk (50 MiB) of a larger
    file, so it cannot tell apart two such files that differ in between.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _own_hash(sha256: str) -> str:
    """A LaminDB ``hash`` value (22 characters) made from a SHA-256."""
    from lamindb_setup.core.hashing import HASH_LENGTH

    encoded = base64.urlsafe_b64encode(bytes.fromhex(sha256)).decode().strip("=")
    return encoded[:HASH_LENGTH]


def _with_sha256(meta: Dict[str, Any], sha256: Optional[str]) -> Dict[str, Any]:
    return meta if sha256 is None else {**meta, "sha256": sha256}


def _recorded_sha256(record: Any) -> Optional[str]:
    lobster = (record.extra_data or {}).get("lobster")
    value = lobster.get("sha256") if isinstance(lobster, dict) else None
    return value if isinstance(value, str) else None


def _holds_sha(record: Any, path: Path, sha256: str) -> bool:
    """Whether ``record`` is of the bytes of ``path``, whose SHA-256 is ``sha256``.

    A record without a SHA-256 of its own is judged by LaminDB's hash.
    """
    recorded = _recorded_sha256(record)
    if recorded is not None:
        return recorded == sha256
    return record.hash == _lamin_hash(path)


def _version(dist: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:
        return None


class _CurrentStderr:
    """A stream that writes to whatever ``sys.stderr`` is at the time of the write.

    A handler holding the stream object itself would write to a closed file once
    the caller swaps ``sys.stderr`` out (a test runner, a redirect). With no stderr,
    or a closed one, the message is lost: ``logging`` would otherwise raise out of
    LaminDB's log call, into the save that logged it.
    """

    def write(self, text: str) -> int:
        try:
            return sys.stderr.write(text)
        except (AttributeError, ValueError, OSError):
            return 0

    def flush(self) -> None:
        try:
            sys.stderr.flush()
        except (AttributeError, ValueError, OSError):
            pass


def _log_to_stderr(lamindb_setup: Any) -> None:
    """Send LaminDB's log lines to stderr: its logger writes to stdout by default.

    stdout belongs to the program (``--json`` output, the TUI protocol); the
    catalog must not add lines to it.
    """
    for handler in lamindb_setup.logger.handlers:
        if type(handler) is logging.StreamHandler:  # not a FileHandler
            handler.setStream(_CurrentStderr())


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)
