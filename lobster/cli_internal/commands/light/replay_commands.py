"""``lobster replay --verify``: re-run an exported notebook and check what it writes.

The notebook is executed with papermill (``NotebookExecutor``) in a folder of its
own, ``<workspace>/replays/<id>/``. Its output is compared, under a
``ComparisonPolicy``, with the file the session saved, and the result is a
``ReplayReport`` (``lobster.testing.replay_contract``) written next to the output.
The exit code follows the verdict.

A replay says MATCH only if every output the notebook wrote was compared with a file
the session saved, each of those files still holds the bytes recorded when the session
saved it, every slot of the policy was compared, and the notebook changed nothing it
should only read (``data/``, its input file, itself). The notebook is the user's own
code, run with the user's permissions: Lobster does not stop it writing anywhere, it
detects a change to those files afterwards and refuses to call the replay a match.
The kernel starts in the replay folder, so a relative write lands there.

With the LaminDB catalog on, the replay is also recorded: a Run of the notebook's
Transform, initiated by the session that exported it, with the verdict and the
report attached; its files go to ``replays/<run uid>/`` and the reference is what
the catalog recorded for that session. A catalog problem is a warning on stderr
and never changes the verdict. Core never imports LaminDB: only ``LaminCatalog``
does, and only when the catalog is on.

The notebook's save cell names its output ``<output_prefix>_<stem>.h5ad``; the
reference for it is the session's ``<stem>.h5ad``, in the workspace's ``data/``
folder or the workspace root (or the catalog's record of it). Heavy imports
(anndata, papermill, LaminDB) happen inside the command, so registering it costs
nothing at startup.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import typer
from typer.core import TyperCommand

from lobster.i18n import h
from lobster.testing.replay_contract import (
    Claim,
    ComparisonPolicy,
    ContractError,
    Exclusion,
    Execution,
    Preflight,
    Refusal,
    RefusalOrigin,
    ReplayReport,
    SlotStatus,
    Verdict,
    VerificationStatus,
    aggregate_verdict,
    canonical_sha256,
    split_path,
)

BACKEND = "lobster-notebook"
REPLAYS_DIR = "replays"
# One code per verdict, for CI and benchmarks to gate on. 2 is what a usage error
# exits with, on the whole CLI, so no verdict uses it; a notebook that does not run
# is not a MISMATCH either.
EXIT_USAGE = 2
EXIT_NOTEBOOK_FAILED = 5
EXIT_MATCH_CUSTOM_POLICY = 6
EXIT_INTERRUPTED = 130
DEFAULT_TIMEOUT = 3600  # seconds one cell may run
EXIT_CODES = {
    Verdict.MATCH: 0,
    Verdict.MISMATCH: 1,
    Verdict.NOT_COMPARED: 3,
    Verdict.REFUSED: 4,
}
NOTEBOOK_FAILED = (
    "NOTEBOOK_FAILED"  # the verdict word of exit 5 (the report says ERROR)
)
MATCH_CUSTOM_POLICY = "MATCH_CUSTOM_POLICY"  # a match under a policy of the user's own
_SHA256 = re.compile(r"[0-9a-f]{64}")
OUTPUT_PREFIX = "out"

# What a notebook cannot give back: the session's own bookkeeping, written by the
# data manager and not by an analysis step (the same two paths and reason as the
# replay harness's own policy, tests/integration/notebook_replay/policies/s1.json).
SESSION_RECORDS = {
    "uns/provenance": (
        "The session's provenance record, written by the data manager and not by "
        "any analysis step. A notebook does not write its own provenance record."
    ),
    "uns/lineage": (
        "The session's record of which modality each object was derived from "
        "(session bookkeeping, not analysis output). A notebook has no modality "
        "names."
    ),
}

_WORKSPACE_HELP = "Workspace directory. Default: ./.lobster_workspace"
_JSON_HELP = "Output results as JSON (machine-readable)"

Note = Tuple[str, Dict[str, Any]]


try:  # newer typer ships its own click, with its own exception classes
    from typer._click.exceptions import UsageError as _TyperUsageError

    _USAGE_ERRORS: Tuple[type, ...] = (click.UsageError, _TyperUsageError)
except ImportError:
    _USAGE_ERRORS = (click.UsageError,)


class JsonUsageCommand(TyperCommand):
    """A usage error under ``--json`` leaves one JSON document on stdout, rc 2."""

    def parse_args(self, ctx, args):
        given = list(args)  # the parser consumes ``args``
        try:
            return super().parse_args(ctx, args)
        except _USAGE_ERRORS as exc:
            if "--json" not in given:
                raise
            typer.echo(exc.format_message(), err=True)
            typer.echo(
                json.dumps(
                    {
                        "verdict": None,
                        "exit_code": EXIT_USAGE,
                        "error": exc.format_message(),
                    }
                )
            )
            ctx.exit(EXIT_USAGE)


class Refused(Exception):
    """The command cannot even bind a report (no notebook, no input, bad policy)."""


@dataclass
class Outcome:
    report: ReplayReport
    replay_dir: Path
    notes: List[Note] = field(default_factory=list)
    custom_match: bool = False  # MATCH, but only under the user's own policy
    custom_policy: Dict[str, List[str]] = field(default_factory=dict)  # what it covers

    @property
    def report_file(self) -> Path:
        return self.replay_dir / "replay_report.json"

    @property
    def failed(self) -> bool:
        """The notebook did not run to its end (an error or the time limit)."""
        return self.report.execution in (Execution.ERROR, Execution.TIMEOUT)

    @property
    def label(self) -> str:
        if self.failed:
            return NOTEBOOK_FAILED
        return MATCH_CUSTOM_POLICY if self.custom_match else self.report.verdict.value

    @property
    def code(self) -> int:
        if self.failed:
            return EXIT_NOTEBOOK_FAILED
        if self.custom_match:
            return EXIT_MATCH_CUSTOM_POLICY
        return EXIT_CODES[self.report.verdict]


@dataclass(frozen=True)
class MatchFacts:
    """What the check established. ``None`` is "not established"."""

    comparison: Optional[Verdict] = None  # every saved output slot, compared
    record_complete: Optional[bool] = None  # every export-record entry has a hash
    input_disjoint: Optional[bool] = None  # --input is no saved result of the session
    loaded_is_input: Optional[bool] = None  # --input is the one file the session loaded
    outputs_compared: Optional[bool] = None  # at least one saved output was compared
    default_policy: Optional[bool] = None  # no policy of the user's own
    no_writes: Optional[bool] = None  # the notebook changed nothing it should read
    references_intact: Optional[bool] = None  # each saved file holds its recorded bytes


_FACTS = tuple(f.name for f in fields(MatchFacts))
_VIOLATIONS_REFUSE = (
    "input_disjoint",
    "loaded_is_input",
    "no_writes",
    "references_intact",
)


def _grant_match(facts: Any) -> Tuple[str, List[str]]:
    """The only place that returns MATCH. ``(verdict word, the facts behind it)``.

    It starts from "not a match" and flips only when every fact is present and
    true. A violated fact refuses; a missing one is not compared; a different
    fact set than the one known here never matches.
    """
    if type(facts) is not MatchFacts or tuple(vars(facts)) != _FACTS:
        return Verdict.NOT_COMPARED.value, ["facts"]
    values = vars(facts)
    refused = [name for name in _VIOLATIONS_REFUSE if values[name] is False]
    if refused:
        return Verdict.REFUSED.value, refused
    if values["loaded_is_input"] is None:  # the input is not tied to the loaded data
        return Verdict.NOT_COMPARED.value, ["loaded_is_input"]
    if values["comparison"] is Verdict.MISMATCH:
        return Verdict.MISMATCH.value, ["comparison"]
    unproven = []
    for name in _FACTS:
        value = values[name]
        if name == "comparison":
            established = value is Verdict.MATCH
        elif name == "default_policy":
            established = value is not None  # False only makes the match "custom"
        else:
            established = value is True
        if not established:
            unproven.append(name)
    if unproven:
        return Verdict.NOT_COMPARED.value, unproven
    if values["default_policy"] is False:
        return MATCH_CUSTOM_POLICY, []
    return Verdict.MATCH.value, []


# --- bindings ---------------------------------------------------------------


def _checkout_root() -> Path:
    return Path(__file__).resolve().parents[4]


@lru_cache(maxsize=1)
def _source_revision() -> Tuple[str, bool]:
    """(commit, dirty) of the Lobster source that runs, for the report's bindings.

    # Decision (Nov 2026): the contract wants a full commit SHA, and an
    # installed wheel has none. Without a git checkout this is a fingerprint of the
    # version string, reported as dirty, so it never reads as a clean commit.
    Replace this helper when the contract says what an installed release binds to.
    """
    root = _checkout_root()
    if (root / ".git").exists():
        try:
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
            changed = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "-uno"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
            if len(head) == 40:
                return head, bool(changed)
        except (OSError, subprocess.SubprocessError):
            pass
    from lobster.version import __version__

    return hashlib.sha1(f"lobster-ai=={__version__}".encode()).hexdigest(), True


def _bindings(
    *,
    notebook_sha: str,
    record: Dict[str, Any],
    roots: Dict[str, str],
    policy: ComparisonPolicy,
    params: Dict[str, str],
) -> Dict[str, Any]:
    """The report's bindings, all recomputed from files, none taken from the notebook.

    # Decision (Nov 2026): the harness binds a captured provenance file and a kernel
    # receipt. A user run has neither, so the recipe is the notebook's own
    # ``metadata.lobster`` record and the environment is the package list of the
    # process that ran the command (``environment_text``, as the catalog records it).
    """
    from lobster.core.catalog.environment import environment_text
    from lobster.version import __version__

    revision, dirty = _source_revision()
    environment = environment_text(__version__).encode("utf-8")
    return {
        "notebook_sha256": notebook_sha,
        "recipe_sha256": canonical_sha256(record),
        "root_sha256": roots,
        "policy_sha256": policy.digest(),
        "params_sha256": canonical_sha256({"papermill": params, "tools": {}}),
        "source_revision": revision,
        "source_dirty": dirty,
        "env_receipt_sha256": hashlib.sha256(environment).hexdigest(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- policy -----------------------------------------------------------------


def _default_policy(names: List[str]) -> ComparisonPolicy:
    """Exact values; the session's own records are not expected back.

    # Decision (Nov 2026): no tolerance until the noise-floor experiments say which
    # slot may move by how much. ``--policy`` replaces this whole policy.
    """
    excluded = tuple(
        Exclusion(f"files/{name}/{path}", why)
        for name in names
        for path, why in SESSION_RECORDS.items()
    )
    return ComparisonPolicy(required=("files",), excluded=excluded)


def _load_policy(path: Optional[Path]) -> Optional[ComparisonPolicy]:
    if path is None:
        return None
    try:
        return ComparisonPolicy.from_json(path.read_text())
    except (OSError, ContractError) as exc:
        raise Refused(
            h("Policy {path} is not usable: {reason}").format(path=path, reason=exc)
        ) from None


# --- the check --------------------------------------------------------------


def _notebook_record(notebook: Path) -> Tuple[str, Dict[str, Any]]:
    """The notebook's SHA-256 and its ``metadata.lobster`` record."""
    try:
        data = notebook.read_bytes()
    except OSError:
        raise Refused(h("Notebook not found: {path}").format(path=notebook)) from None
    try:
        nb = json.loads(data)
        record = nb["metadata"].get("lobster", {})
        if not isinstance(nb["cells"], list) or not isinstance(record, dict):
            raise ValueError("no cells")
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        raise Refused(
            h("{path} is not a Jupyter notebook: {reason}").format(
                path=notebook, reason=exc
            )
        ) from None
    return hashlib.sha256(data).hexdigest(), record


def _read_h5ad(path: Path):
    """An h5ad file as the notebook's kernel would read it, without file-layout names.

    ``lobster`` switches pandas' string inference off, the notebook's kernel keeps
    pandas' default, and the data manager and plain anndata store the index under
    different names. Neither says anything about the data, yet each would show as
    a difference between the saved file and the notebook's output.
    """
    import anndata
    import pandas as pd

    if int(pd.__version__.split(".")[0]) >= 3:
        with pd.option_context("future.infer_string", True):
            adata = anndata.read_h5ad(path)
    else:
        adata = anndata.read_h5ad(path)
    adata.obs.index.name = None
    adata.var.index.name = None
    return adata


def _manifest(notebook: Path) -> Dict[str, Any]:
    """The ``data_integrity_manifest`` of the export record, as written.

    The record is a JSON block in a markdown cell. Empty when there is no record or
    it cannot be read.
    """
    try:
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    for cell in cells:
        source = cell.get("source", "") if isinstance(cell, dict) else ""
        if isinstance(source, list):
            source = "".join(source)
        if "data_integrity_manifest" not in source:
            continue
        found = re.search(r"```json\s*(\{.*?\})\s*```", source, re.DOTALL)
        try:
            manifest = json.loads(found.group(1))["data_integrity_manifest"]
            return dict(manifest) if isinstance(manifest, dict) else {}
        except (AttributeError, ValueError, KeyError, TypeError):
            return {}
    return {}


def _record_entries(notebook: Path) -> Dict[str, Any]:
    """Every saved-file entry of the export record: file name -> hash or a word."""
    files = _manifest(notebook).get("input_files")
    return dict(files) if isinstance(files, dict) else {}


def _loaded_files(notebook: Path) -> Optional[List[Dict[str, Any]]]:
    """The files the session loaded, from the record; None for a record without roles."""
    roles = _manifest(notebook).get("input_roles")
    if not isinstance(roles, dict) or roles.get("version") != 2:
        return None
    loaded = roles.get("loaded")
    return list(loaded) if isinstance(loaded, list) else None


def _loaded_is_input(
    loaded: Optional[List[Dict[str, Any]]], input_data: Path, input_sha: str
) -> Tuple[Optional[bool], Optional[Note], Dict[str, Any]]:
    """Is ``--input`` the one thing the session loaded?
    ``(fact, why not established, the loaded entry it is)``.

    A loaded file is the input when its SHA-256 is the input's; a loaded object (a
    download hands one over) when its content digest is the input's. ``True`` when
    every loaded entry is the input (one loaded twice counts once); ``False`` when
    none is; ``None`` when the record cannot say.
    """
    if loaded is None:
        return (
            None,
            (
                "The export record predates input roles v2; re-export the notebook.",
                {},
            ),
            {},
        )
    from lobster.core.provenance import content_digest as digest_module

    if not digest_module.IN_MEMORY_INPUTS and any(
        isinstance(e, dict) and ("content_sha256" in e or e.get("in_memory"))
        for e in loaded
    ):
        return None, ("In-memory input not anchorable in this version.", {}), {}
    keys = [_loaded_key(e) for e in loaded]
    if not keys or None in keys:
        return None, ("The export record names no file the session loaded.", {}), {}
    digest = None
    matches = set()
    for kind, value in set(keys):
        if kind == "file":
            same = value == input_sha
        else:
            if digest is None:
                digest = _content_digest_of(input_data)
            same = value == digest
        if same:
            matches.add((kind, value))
    if not matches:
        return False, None, {}
    if len(matches) < len(set(keys)):
        return (
            None,
            (
                "The session loaded {count} different files; one --input cannot stand "
                "for all of them.",
                {"count": len(set(keys))},
            ),
            {},
        )
    entry = next(e for e, k in zip(loaded, keys) if k in matches)
    return True, None, entry


def _session_writer_used(executed: Path) -> Optional[bool]:
    """False when a cell of the executed notebook printed the fallback line.

    None if the executed notebook cannot be read.
    """
    from lobster.core.notebooks.exporter import SESSION_WRITER_FALLBACK_LINE

    try:
        cells = json.loads(executed.read_text("utf-8")).get("cells", [])
    except (OSError, ValueError):
        return None
    for cell in cells:
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            text = "".join(text) if isinstance(text, list) else str(text)
            if (
                output.get("output_type") == "stream"
                and SESSION_WRITER_FALLBACK_LINE in text
            ):
                return False
    return True


def _loaded_key(entry: Any) -> Optional[Tuple[str, str]]:
    """``("file", sha256)`` or ``("object", content digest)``; None if neither."""
    if not isinstance(entry, dict):
        return None
    for kind, field_name in (("file", "sha256"), ("object", "content_sha256")):
        value = entry.get(field_name)
        if kind == "object" and entry.get("digest") != "v2":
            continue
        if isinstance(value, str) and _SHA256.fullmatch(value):
            return kind, value
    return None


def _content_digest_of(path: Path) -> Optional[str]:
    from lobster.core.provenance.content_digest import content_digest

    try:
        return content_digest(_read_h5ad(path))
    except Exception:  # not an h5ad file: it matches no loaded object
        return None


def _recorded_hashes(notebook: Path) -> Dict[str, str]:
    """SHA-256 of each saved file the export record has a hash for."""
    return {
        k: v
        for k, v in _record_entries(notebook).items()
        if isinstance(v, str) and _SHA256.fullmatch(v)
    }


def _workspace_reference(
    workspace: Path, stem: str, recorded: Dict[str, str]
) -> Tuple[Optional[Path], str]:
    """``(file, "ok")``, ``(None, "changed")``, ``(None, "unrecorded")`` or ``(None, "none")``.

    The file is the workspace's ``<stem>.h5ad``. It is the reference only if it
    still holds the bytes the export record has for it; no record, no check.

    # A file with no recorded hash is "unrecorded" (not compared), never an
    # unchecked MATCH. Flip it here to trust an unrecorded file.
    """
    for folder in (workspace / "data", workspace):
        candidate = folder / f"{stem}.h5ad"
        if candidate.is_file():
            want = recorded.get(f"{stem}.h5ad")  # the record is keyed by file name
            if want is None:
                return None, "unrecorded"
            return (
                (candidate, "ok") if _sha256(candidate) == want else (None, "changed")
            )
    return None, "none"


class _Unreadable(Exception):
    """A file the check watches cannot be read; ``args[0]`` names it."""


DATA_SUFFIXES = (".h5ad", ".h5mu")


def _same_content(saved: Path, other: Path, name: str) -> bool:
    """Do the two files hold the same data, under the default policy?

    A file that cannot be read for the comparison is not the same.
    """
    from lobster.testing.replay_compare import compare

    try:
        slots = compare(
            {"files": {name: _read_h5ad(saved)}},
            {"files": {name: _read_h5ad(other)}},
            _default_policy([name]),
        )
    except Exception:
        return False
    return aggregate_verdict(slots).value == "MATCH"


def _saved_copy(
    saved: Path, loaded: Dict[str, Any], input_data: Path, name: str
) -> bool:
    """Is ``saved`` the loaded input as the session's load left it?

    For a file the session loaded through an adapter: the saved file equals that
    adapter run on ``--input``, under the default policy. For a loaded object (no
    adapter ran): it equals the input. An adapter that is not recorded, or that fails,
    leaves it not a copy.
    """
    if loaded.get("content_sha256"):
        return _same_content(saved, input_data, name)
    adapter = loaded.get("adapter")
    if not isinstance(adapter, str):
        return False
    import tempfile

    try:
        from lobster.core.adapters.factory import create_default_adapters

        result = create_default_adapters()[adapter].from_source(input_data)
        with tempfile.TemporaryDirectory() as folder:
            produced = Path(folder) / "loaded.h5ad"
            result.write_h5ad(produced)
            return _same_content(saved, produced, name)
    except Exception:  # the adapter cannot be run here: not excused
        return False


def _input_is_a_saved_result(
    input_data: Path,
    input_sha: str,
    references: Dict[str, Path],
) -> List[str]:
    """Names of compared saved files the input equals, by bytes or by content.

    A replay that starts from a result of the session has not replayed the steps
    that made it. Content is compared under the default policy, whatever policy the
    user gave (a narrow one would call different files equal).
    """
    return [
        name
        for name, saved in sorted(references.items())
        if _sha256(saved) == input_sha or _same_content(saved, input_data, name)
    ]


def _outputs(replay_dir: Path) -> List[Path]:
    """The ``out_<name>.h5ad`` files a replay wrote (any case of the suffix)."""
    return sorted(
        p
        for p in replay_dir.iterdir()
        if p.is_file()
        and p.name.startswith(f"{OUTPUT_PREFIX}_")
        and p.suffix.lower() == ".h5ad"
    )


def _watched(workspace: Path, skip: Path, *files: Path) -> Dict[str, str]:
    """SHA-256 of what a replay must leave alone, by path.

    That is every file under ``data/``, every ``.h5ad``/``.h5mu`` anywhere in the
    workspace outside ``.lamindb/`` and ``skip`` (this run's folder), and ``files``.
    """
    root = workspace.resolve()
    skipped = skip.resolve()
    data_root = (root / "data").resolve()
    paths = {p.resolve() for p in files if p.is_file()}
    seen = {root}  # real folders already walked: a link back into one is not walked
    # Links are followed (data/ may be a link to a bigger disk); names stay lexical.
    for folder, dirs, names in os.walk(root, followlinks=True):
        here = Path(folder)
        keep = []
        for d in dirs:
            real = (here / d).resolve()
            if d != ".lamindb" and real != skipped and real not in seen:
                seen.add(real)
                keep.append(d)
        dirs[:] = keep
        real_here = here.resolve()
        in_data = real_here == data_root or data_root in real_here.parents
        paths |= {
            here / n for n in names if in_data or n.lower().endswith(DATA_SUFFIXES)
        }
    hashes = {}
    for path in sorted(paths):
        try:
            hashes[str(path)] = _sha256(path)
        except OSError as exc:
            try:
                name = path.relative_to(root).as_posix()
            except ValueError:
                name = path.name
            raise _Unreadable(name) from exc
    return hashes


def _changes(before: Dict[str, str], after: Dict[str, str], workspace: Path):
    """The files that differ, were added or were removed, named from the workspace."""
    root = workspace.resolve()

    def name(path: str) -> str:
        try:
            return Path(path).relative_to(root).as_posix()
        except ValueError:
            return Path(path).name

    return sorted(
        name(k) for k in before.keys() | after.keys() if before.get(k) != after.get(k)
    )


@contextlib.contextmanager
def _working_in(folder: Path):
    """The kernel starts in the process's directory: make that the replay folder."""
    previous = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(previous)


def _not_compared(slots, lines: Tuple[str, ...]):
    """Slots that cannot say MATCH while some output was left out of the comparison."""
    return tuple(
        (
            replace(slot, status=SlotStatus.NOT_COMPARED, not_compared=lines)
            if slot.status is SlotStatus.MATCH
            else replace(slot, not_compared=slot.not_compared + lines)
        )
        for slot in slots
    )


def verify_notebook(
    notebook: Path,
    workspace: Path,
    input_data: Optional[Path],
    policy_file: Optional[Path] = None,
    *,
    catalog: Optional["_ReplayRecord"] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Outcome:
    """Run ``notebook`` on ``input_data`` and compare; see the module docstring.

    ``catalog`` (the catalog side of the replay) is begun once the inputs are valid and ended with
    the outcome. Raises ``Refused`` when no report can be bound.
    """
    # The kernel runs in the replay folder: every path is made absolute first.
    notebook, workspace = notebook.resolve(), workspace.resolve()
    input_data = input_data.resolve() if input_data is not None else None
    policy_file = policy_file.resolve() if policy_file is not None else None
    if input_data is None:
        raise Refused(h("Give the notebook's input file with --input."))
    notebook_sha, record = _notebook_record(notebook)
    if not input_data.is_file():
        raise Refused(h("Input file not found: {path}").format(path=input_data))
    roots = {input_data.name: _sha256(input_data)}
    user_policy = _load_policy(policy_file)
    # Machine paths stay out of the binding: the same replay binds the same way.
    params = {"input_data": input_data.name, "output_prefix": OUTPUT_PREFIX}
    run_uid = catalog.begin(notebook, workspace, params) if catalog else None
    entries = _record_entries(notebook)
    recorded = _recorded_hashes(notebook)
    find = (
        (lambda stem: catalog.reference(stem, recorded))
        if run_uid and catalog.has_session
        else (lambda stem: _workspace_reference(workspace, stem, recorded))
    )

    replay_dir = workspace / REPLAYS_DIR / (run_uid or secrets.token_hex(8))
    replay_dir.mkdir(parents=True)
    outcome = Outcome(report=None, replay_dir=replay_dir)  # type: ignore[arg-type]

    def report(policy, **fields) -> ReplayReport:
        base = dict(
            backend=BACKEND,
            scenario_id=notebook.stem or "notebook",
            requested_mode="replay",
            claim_requested=Claim.HIST,
            claim_achieved=None,  # not isolated: this run certifies nothing
            bindings=_bindings(
                notebook_sha=notebook_sha,
                record=record,
                roots=roots,
                policy=policy,
                params=params,
            ),
        )
        built = ReplayReport(**{**base, **fields})
        built.validate(policy)
        return built

    def finish(built: ReplayReport) -> Outcome:
        outcome.report = built
        outcome.report_file.write_text(_report_json(outcome) + "\n")
        if run_uid:
            catalog.end(outcome)
        return outcome

    def refused(policy, reason: str, step: str) -> Outcome:
        """A MATCH cannot be said: what the replay stands on is not what it was."""
        return finish(
            report(
                policy,
                verification_status=VerificationStatus.NOT_RUN,
                preflight=Preflight.ADMITTED,
                execution=Execution.REFUSED_AT_RUNTIME,
                verdict=Verdict.REFUSED,
                refusal=Refusal(reason, step, RefusalOrigin.PRODUCT),
            )
        )

    base_policy = user_policy or _default_policy([])
    outputs = replay_dir.resolve() / f"{OUTPUT_PREFIX}"
    executed = replay_dir.resolve() / "executed.ipynb"
    from lobster.core.notebooks.executor import NotebookExecutor

    watched = (notebook.resolve(), input_data.resolve())
    unreadable = lambda name: refused(  # noqa: E731
        base_policy,
        h("A file the check watches cannot be read: {names}").format(names=name),
        name,
    )
    try:
        before = _watched(workspace, replay_dir, *watched)
    except _Unreadable as exc:
        return unreadable(exc.args[0])
    try:
        # The executor logs to stdout; stdout is the result of this command.
        with _working_in(replay_dir), contextlib.redirect_stdout(sys.stderr):
            result = NotebookExecutor(None).execute(
                notebook.resolve(),
                input_data.resolve(),
                {"output_prefix": str(outputs)},
                executed,
                timeout=timeout or None,
            )
    except Exception as exc:  # e.g. a parameters cell that is not Python
        result = {
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }

    status = result.get("status")
    if status == "validation_failed":
        reason = "; ".join(result.get("errors") or ["the notebook was not run"])
        return finish(
            report(
                base_policy,
                verification_status=VerificationStatus.NOT_RUN,
                preflight=Preflight.REFUSED,
                execution=Execution.NOT_RUN,
                verdict=Verdict.REFUSED,
                refusal=Refusal(reason, None, RefusalOrigin.HARNESS),
            )
        )
    # Detected, not prevented: the notebook is the user's code.
    try:
        after = _watched(workspace, replay_dir, *watched)
    except _Unreadable as exc:
        return unreadable(exc.args[0])
    changed = _changes(before, after, workspace)
    if changed:
        return refused(
            base_policy,
            h("The notebook changed files it should only read: {names}").format(
                names=", ".join(changed)
            ),
            changed[0],
        )
    if status != "success":
        timed_out = "Timeout" in str(result.get("error_type"))
        if timed_out:
            outcome.notes.append(
                (
                    "The notebook ran past the time limit ({seconds} seconds for "
                    "one cell).",
                    {"seconds": timeout},
                )
            )
        else:
            outcome.notes.append(
                (
                    "The notebook stopped with {error_type}: {error}",
                    {
                        "error_type": result.get("error_type"),
                        "error": result.get("error"),
                    },
                )
            )
        return finish(
            report(
                base_policy,
                verification_status=VerificationStatus.NOT_RUN,
                preflight=Preflight.ADMITTED,
                execution=Execution.TIMEOUT if timed_out else Execution.ERROR,
                verdict=Verdict.NOT_COMPARED,
            )
        )

    expected: Dict[str, Any] = {}
    actual: Dict[str, Any] = {}
    left_out: List[str] = []  # outputs that were not compared
    edited: List[str] = []  # outputs whose saved file is not what the session saved
    references: Dict[str, Path] = {}  # compared output name -> the saved file
    produced = _outputs(replay_dir)
    if not produced:
        outcome.notes.append(("The notebook wrote no .h5ad file.", {}))
    for path in produced:
        stem = path.name[len(OUTPUT_PREFIX) + 1 : -len(".h5ad")]
        name = f"{stem}.h5ad"
        try:
            split_path(f"files/{name}")  # a policy path names it literally
            saved, state = find(stem)
        except ContractError:
            saved, state = None, "none"
        if state == "changed":
            edited.append(name)
            continue
        if saved is None:
            left_out.append(name)
            if state == "unrecorded":
                outcome.notes.append(
                    (
                        "{name}: the export record has no hash for the saved file, "
                        "so it cannot be checked against what the session saved.",
                        {"name": name},
                    )
                )
            else:
                outcome.notes.append(
                    (
                        "{name}: the session has no saved file to compare it with.",
                        {"name": name},
                    )
                )
            continue
        try:
            pair = (_read_h5ad(saved), _read_h5ad(path))
        except Exception as exc:  # a file that cannot be read cannot be compared
            left_out.append(name)
            outcome.notes.append(
                (
                    "{name}: could not be read for the comparison ({error}).",
                    {"name": name, "error": exc},
                )
            )
            continue
        expected[name], actual[name] = pair
        references[name] = saved

    input_sha = roots[input_data.name]
    loaded = _loaded_files(notebook)
    loaded_state, loaded_note, loaded_entry = _loaded_is_input(
        loaded, input_data, input_sha
    )
    if loaded_note:
        outcome.notes.append(loaded_note)
    input_name = _input_is_a_saved_result(input_data, input_sha, references)
    unproduced = sorted(n for n in recorded if n not in expected and n not in left_out)
    same_file: List[str] = []  # the loaded input file, saved under the session
    copies: List[str] = []  # saved files that are the loaded input itself
    for name in unproduced:
        saved, state = find(name[: -len(".h5ad")])
        if loaded_state is not True or state != "ok":
            continue
        if loaded_entry.get("sha256") == input_sha == recorded.get(name):
            same_file.append(name)  # the loaded file itself, byte for byte
        elif _saved_copy(saved, loaded_entry, input_data, name):
            copies.append(name)
    if same_file:
        unproduced = [n for n in unproduced if n not in same_file]
        outcome.notes.append(
            (
                "Loaded input file, not compared: {names}",
                {"names": ", ".join(same_file)},
            )
        )
    if copies:
        unproduced = [n for n in unproduced if n not in copies]
        outcome.notes.append(
            (
                "Saved copy of the input, not compared: {names}",
                {"names": ", ".join(copies)},
            )
        )
    if unproduced:  # saved by the session, never written by the notebook
        left_out.extend(unproduced)
        outcome.notes.append(
            (
                "Saved by the session but not produced by the notebook: {names}",
                {"names": ", ".join(unproduced)},
            )
        )
    without_hash = sorted(k for k in entries if k not in recorded)
    loaded_names = {
        f"{e.get('modality')}.h5ad" for e in loaded or () if isinstance(e, dict)
    }
    loaded_only = [k for k in without_hash if k in loaded_names and loaded_state]
    # "not_saved": the session never saved that modality; there is nothing to hash
    without_hash = [
        k for k in without_hash if k not in loaded_only and entries[k] != "not_saved"
    ]
    if loaded_only:  # the modality is the loaded input; it has no saved file to hold
        outcome.notes.append(
            (
                "Loaded input, no saved file: {names}",
                {"names": ", ".join(loaded_only)},
            )
        )
    if without_hash:  # an entry the check cannot hold a saved file to
        outcome.notes.append(
            (
                "The export record has no hash for: {names}",
                {"names": ", ".join(without_hash)},
            )
        )
    written = {path.name for path in produced} | {
        "executed.ipynb",
        "replay_report.json",
    }
    others = sorted(
        path.relative_to(replay_dir).as_posix()
        for path in replay_dir.rglob("*")
        if path.is_file() and not (path.parent == replay_dir and path.name in written)
    )
    for name in others:  # data the session did not produce: never a MATCH
        if name.lower().endswith(DATA_SUFFIXES):
            left_out.append(name)
            outcome.notes.append(
                (
                    "{name}: written by the notebook; the session has no saved "
                    "file to compare it with.",
                    {"name": name},
                )
            )
    listed = [name for name in others if not name.lower().endswith(DATA_SUFFIXES)]
    if listed:
        outcome.notes.append(
            (
                "Other files the notebook wrote: {names}",
                {"names": ", ".join(listed[:20])},
            )
        )

    if edited:
        return refused(
            base_policy,
            h("The saved file changed after the session: {names}").format(
                names=", ".join(edited)
            ),
            edited[0],
        )

    if input_name:  # the replay did not start from what the session started from
        return refused(
            base_policy,
            h("The input is a saved result of the session: {names}").format(
                names=", ".join(input_name)
            ),
            input_name[0],
        )

    from lobster.testing.replay_compare import compare

    policy = user_policy or _default_policy(sorted(expected))
    slots = compare({"files": expected}, {"files": actual}, policy) if actual else ()
    if left_out:
        slots = _not_compared(
            slots, tuple(f"{name}: not compared with a saved file" for name in left_out)
        )
    comparison = aggregate_verdict(slots)
    writer_used = _session_writer_used(executed)
    if writer_used is False:
        outcome.notes.append(("The notebook wrote without the session writer.", {}))
    word, reasons = _grant_match(
        MatchFacts(
            comparison=comparison,
            record_complete=bool(entries) and not without_hash,
            input_disjoint=True,  # a saved result as the input was refused above
            loaded_is_input=loaded_state,
            outputs_compared=any(slot.compared for slot in slots),
            default_policy=user_policy is None,
            no_writes=True,  # a change was refused above
            references_intact=True,  # an edited reference was refused above
        )
    )
    if word == Verdict.REFUSED.value and "loaded_is_input" in reasons:
        names = sorted(e.get("name", "?") for e in loaded)
        return refused(
            base_policy,
            h("The input is not a file the session loaded: {names}").format(
                names=", ".join(map(str, names))
            ),
            names[0],
        )
    verdict = Verdict.MATCH if word == MATCH_CUSTOM_POLICY else Verdict(word)
    if verdict is Verdict.NOT_COMPARED and comparison in (
        Verdict.MATCH,
        Verdict.MISMATCH,
    ):
        # the slots compared, but the input facts keep the verdict from being
        # granted either way: say so in the slots
        lines = tuple(f"not established: {r}" for r in reasons)
        slots = (
            _not_compared(slots, lines)
            if comparison is Verdict.MATCH
            else tuple(  # the differences are not reported as the input is unanchored
                replace(s, status=SlotStatus.NOT_COMPARED, diffs=(), not_compared=lines)
                for s in slots
            )
        )
    if verdict in (Verdict.MATCH, Verdict.MISMATCH):
        verification = VerificationStatus.CHECKED
    elif slots:
        verification = VerificationStatus.INCOMPLETE
    else:
        verification = VerificationStatus.NOT_RUN
    if word == MATCH_CUSTOM_POLICY:
        outcome.custom_match = True
        outcome.custom_policy = {
            "required": list(policy.required),
            "excluded": [e.path for e in policy.excluded],
        }
        outcome.notes.append(
            (
                "Under a policy of your own: compared {required}; excluded {excluded}.",
                {
                    "required": ", ".join(policy.required),
                    "excluded": ", ".join(e.path for e in policy.excluded) or "nothing",
                },
            )
        )
    return finish(
        report(
            policy,
            verification_status=verification,
            preflight=Preflight.ADMITTED,
            execution=Execution.OK,
            verdict=verdict,
            slots=slots,
            excluded=policy.excluded if slots else (),
        )
    )


# --- the catalog ------------------------------------------------------------


def _notebook_key(notebook: Path, workspace: Path) -> str:
    """The notebook's key in the catalog: its path in the workspace, as the export records it."""
    try:
        return notebook.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return f"notebooks/{notebook.name}"


class _ReplayRecord:
    """The catalog side of a replay. A catalog problem is a warning, never a verdict."""

    def __init__(self, workspace: Path, instance: Optional[str]):
        self.workspace = workspace
        self.instance = instance
        self.engine: Any = None
        self.run: Any = None
        self.closed = False

    @property
    def has_session(self) -> bool:
        """Whether the catalog knows the session that exported the notebook."""
        return self.run is not None and self.run.session_uid is not None

    def _failed(self, exc: Exception) -> None:
        _catalog_failed(exc)

    def begin(
        self, notebook: Path, workspace: Path, params: Dict[str, str]
    ) -> Optional[str]:
        """Open the replay's Run; its uid (the folder name), or None to go on without."""
        from lobster.cli_internal.commands.light.lamin_commands import _open
        from lobster.version import __version__

        try:
            self.engine = _open(self.workspace, instance=self.instance)
            self.run = self.engine.begin_replay(
                key=_notebook_key(notebook, workspace),
                source=notebook.read_text(encoding="utf-8"),
                params=params,
                lobster_version=__version__,
            )
        except Exception as exc:
            self._failed(exc)
            return None
        if self.run is None:
            typer.echo(
                h(
                    "The notebook was not recorded in the catalog; the check goes on "
                    "without it."
                ),
                err=True,
            )
            return None
        return self.run.uid

    def reference(
        self, stem: str, recorded: Dict[str, str]
    ) -> Tuple[Optional[Path], str]:
        try:
            return self.engine.replay_reference_state(stem)
        except Exception as exc:
            self._failed(exc)
            return _workspace_reference(self.workspace, stem, recorded)

    def end(self, outcome: Outcome) -> None:
        report = outcome.report
        self._close(
            "completed" if report.execution == Execution.OK else "errored",
            outcome.label,
            outcome.report_file,
            _outputs(outcome.replay_dir),
        )

    def abort(self, status: str = "errored", verdict: str = NOTEBOOK_FAILED) -> None:
        """Close the Run of a replay that stopped on an unexpected error or a Ctrl-C."""
        if self.run is not None and not self.closed:
            self._close(status, verdict, None, [])

    def _close(self, status, verdict, report_file, outputs) -> None:
        self.closed = True
        try:
            self.engine.end_replay(
                status, verdict=verdict, report_path=report_file, outputs=outputs
            )
        except Exception as exc:
            self._failed(exc)


def _catalog_failed(exc: Exception) -> None:
    typer.echo(
        h(
            "The catalog could not record the replay ({error}); the check goes "
            "on without it."
        ).format(error=f"{type(exc).__name__}: {exc}"[:200]),
        err=True,
    )


def _catalog_record(workspace: Path) -> Optional[_ReplayRecord]:
    """The catalog side of a replay when the catalog is on and has something to attach to."""
    from lobster.cli_internal.commands.light import lamin_commands
    from lobster.core.catalog.factory import resolve_catalog_config

    config = resolve_catalog_config(workspace)
    if not config.enabled or not lamin_commands._engine_installed():
        return None
    if config.mode == "workspace" and lamin_commands._slug(workspace) is None:
        return None  # nothing recorded in this workspace yet
    return _ReplayRecord(workspace, config.instance)


# --- the command ------------------------------------------------------------


def _note_texts(outcome: Outcome, *, translated: bool) -> List[str]:
    return [
        (h(text) if translated else text).format(**fields)
        for text, fields in outcome.notes
    ]


def _report_dict(outcome: Outcome) -> Dict[str, Any]:
    """The report as written; a match under the user's own policy says so in it."""
    report = outcome.report.to_dict()
    if outcome.custom_match:
        report["verdict"] = MATCH_CUSTOM_POLICY
        report["custom_policy"] = outcome.custom_policy
    return report


def _report_json(outcome: Outcome) -> str:
    return json.dumps(_report_dict(outcome), indent=2, sort_keys=True, allow_nan=False)


def _print_result(outcome: Outcome, code: int, as_json: bool) -> None:
    report = outcome.report
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "verdict": outcome.label,
                    "exit_code": code,
                    "report": _report_dict(outcome),
                    "report_file": str(outcome.report_file),
                    "replay_dir": str(outcome.replay_dir),
                    "notes": _note_texts(outcome, translated=False),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    typer.echo(h("Verdict: {verdict}").format(verdict=outcome.label))
    if report.verdict is Verdict.MATCH:
        typer.echo(h("Scope: the files the session saved and the export recorded."))
    if report.refusal is not None:
        typer.echo(h("Refused: {reason}").format(reason=report.refusal.reason))
    for slot in report.slots:
        for line in slot.diffs[:5] + slot.not_compared[:5]:
            typer.echo(f"  {line}")
    for line in _note_texts(outcome, translated=True):
        typer.echo(f"  {line}")
    typer.echo(h("Replay report: {path}").format(path=outcome.report_file))
    typer.echo(h("Replay files: {path}").format(path=outcome.replay_dir))


def replay_command(
    verify: Path = typer.Option(
        ...,
        "--verify",
        help=h("Exported notebook to re-run and check"),
    ),
    input_data: Optional[Path] = typer.Option(
        None,
        "--input",
        help=h("Input data file of the notebook (.h5ad)"),
    ),
    policy: Optional[Path] = typer.Option(
        None,
        "--policy",
        help=h(
            "Comparison policy (JSON). Default: exact values, without the "
            "session's own provenance and lineage records"
        ),
    ),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help=h(_WORKSPACE_HELP)
    ),
    as_json: bool = typer.Option(False, "--json", help=h(_JSON_HELP)),
    timeout: int = typer.Option(
        DEFAULT_TIMEOUT,
        "--timeout",
        min=0,
        help=h(
            "Time limit per cell, in seconds; a cell that runs longer stops the check (0: no limit)"
        ),
    ),
):
    """Re-run an exported notebook and compare its output with the saved file."""
    from lobster.core.runtime.workspace import resolve_workspace

    ws = resolve_workspace(workspace, create=False)
    try:
        record = _catalog_record(ws)
    except Exception as exc:  # a catalog problem is a warning, never a verdict
        _catalog_failed(exc)
        record = None
    try:
        outcome = verify_notebook(
            verify, ws, input_data, policy, catalog=record, timeout=timeout
        )
    except KeyboardInterrupt:
        with contextlib.redirect_stdout(sys.stderr):
            gc.collect()  # a task the kernel client left behind reports itself here
        if record is not None:
            record.abort("aborted", "INTERRUPTED")
        typer.echo(h("Interrupted."), err=True)
        if as_json:
            typer.echo(
                json.dumps(
                    {
                        "verdict": None,
                        "exit_code": EXIT_INTERRUPTED,
                        "error": "interrupted",
                    }
                )
            )
        raise typer.Exit(EXIT_INTERRUPTED)
    except Refused as exc:
        typer.echo(str(exc), err=True)
        if as_json:  # stdout stays one document
            typer.echo(
                json.dumps(
                    {"verdict": None, "exit_code": EXIT_USAGE, "error": str(exc)}
                )
            )
        raise typer.Exit(EXIT_USAGE)
    except Exception as exc:  # whatever else goes wrong is a refusal, not a traceback
        if record is not None:
            record.abort("errored", "REFUSED")
        message = h("The check could not run: {error}").format(
            error=f"{type(exc).__name__}: {exc}"
        )
        typer.echo(message, err=True)
        if as_json:
            typer.echo(
                json.dumps(
                    {
                        "verdict": Verdict.REFUSED.value,
                        "exit_code": EXIT_CODES[Verdict.REFUSED],
                        "error": message,
                    }
                )
            )
        raise typer.Exit(EXIT_CODES[Verdict.REFUSED])
    except BaseException:
        if record is not None:
            record.abort()
        raise
    code = outcome.code
    _print_result(outcome, code, as_json)
    raise typer.Exit(code)
