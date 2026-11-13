"""Where two views get their data: ontology terms and sequences.

Both are filled in by Python when ask_user builds the question, so the LLM
never types ids or letters it could get wrong:

- ``options_from`` (view "ontology_browser"): the supervisor names a public
  ontology and suggests terms by name. Python loads the ontology (downloaded
  once, then kept in ``~/.lobster/ontologies``), turns each name into its
  real term id, and fills ``options`` with the suggestions and the terms
  around them (their path to the top, siblings and children). An unknown name
  is a problem the supervisor fixes. When the ontology can't be loaded (the
  optional ``lobster-ai[vector-search]`` install is missing, or the download
  fails), the question is asked as a plain choice of the suggested names
  instead, with a notice to the user and a note to the supervisor.
- ``sequence_from`` (view "sequence"): the supervisor names a FASTA file and
  record; Python reads the letters, and the file is watched like any data
  file (a change asks again, with the sequence read again).

Ontology loading reuses lobster-vector's download addresses (``OBO_URLS``)
and its OBO reader, obonet.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import ValidationError

from lobster.services.interaction.component_schemas import (
    SEQUENCE_MAX_LENGTH,
    DecisionRequest,
    OptionsFrom,
    Problem,
    SequenceFrom,
    _file_inside,
    _problem,
    _some,
    problems_of,
)

logger = logging.getLogger(__name__)

# Where downloaded ontologies are kept, so each one is downloaded once.
ONTOLOGY_CACHE = Path.home() / ".lobster" / "ontologies"
_DOWNLOAD_TIMEOUT_S = 120

# How many terms one question shows: the suggestions' paths to the top come
# first, then their siblings and children until this many.
NEIGHBOURHOOD_MAX = 200
_SIBLINGS_MAX = 10
_CHILDREN_MAX = 30
_DEPTH_MAX = 40  # a path to the top longer than this is cut

INSTALL_HINT = "lobster-ai[vector-search]"

# The first word of a suggested term's details: screens list these first.
SUGGESTED = "suggested"

_FASTA_BYTES_MAX = 20 * 1024 * 1024


class OntologyUnavailable(Exception):
    """The ontology can't be loaded here (not installed, or not downloadable)."""


# ---------------------------------------------------------------------------
# Ontologies
# ---------------------------------------------------------------------------


class Ontology:
    """A loaded ontology: its is_a tree and an index of term names.

    ``graph`` is a networkx graph as obonet reads it: one node per term
    (with "name", "def", "synonym", "is_obsolete"), and edges from a child
    to its parent keyed by the relation ("is_a", "part_of", ...).
    """

    def __init__(self, name: str, graph: Any):
        self.name = name
        self.graph = graph
        self._index: Dict[str, str] = {}
        exact: Dict[str, str] = {}
        for term, data in graph.nodes(data=True):
            if _obsolete(data):
                continue
            if data.get("name"):
                self._index.setdefault(_key(data["name"]), term)
            for synonym in data.get("synonym") or []:
                found = _EXACT_SYNONYM.match(synonym)
                if found:
                    exact.setdefault(_key(found.group(1)), term)
        for name, term in exact.items():  # a term's own name wins
            self._index.setdefault(name, term)

    def find(self, wanted: str) -> Optional[str]:
        """The id of a term given by its id, exact name or exact synonym."""
        wanted = wanted.strip()
        if wanted in self.graph.nodes and not _obsolete(self.graph.nodes[wanted]):
            return wanted
        return self._index.get(_key(wanted))

    def label(self, term: str) -> str:
        return self.graph.nodes[term].get("name") or term

    def definition(self, term: str) -> Optional[str]:
        found = _QUOTED.match(self.graph.nodes[term].get("def") or "")
        if not found:
            return None
        text = found.group(1)
        return text if len(text) <= 160 else text[:157].rstrip() + "..."

    def parents(self, term: str) -> List[str]:
        """The term's is_a parents, by id."""
        return sorted(
            parent
            for _, parent, relation in self.graph.out_edges(term, keys=True)
            if relation == "is_a"
        )

    def children(self, term: str, parts: bool = False) -> List[str]:
        """The term's is_a children that aren't obsolete, by name; with
        ``parts``, its parts too (part_of: in anatomy the finer terms, e.g.
        liver lobule is part of liver, not a kind of liver)."""
        relations = ("is_a", "part_of") if parts else ("is_a",)
        found = {
            child
            for child, _, relation in self.graph.in_edges(term, keys=True)
            if relation in relations and not _obsolete(self.graph.nodes[child])
        }
        return sorted(found, key=lambda term: self.label(term).casefold())

    def path_to_top(self, term: str) -> List[str]:
        """From the top of the tree down to ``term``, following the first
        is_a parent at each step."""
        path = [term]
        while len(path) < _DEPTH_MAX:
            parents = [p for p in self.parents(path[-1]) if p not in path]
            if not parents:
                break
            path.append(parents[0])
        return path[::-1]

    def neighbourhood(self, suggested: List[str]) -> List[Dict[str, Any]]:
        """The options a screen shows: every suggestion with its path to the
        top, then the suggestions' siblings, kinds and parts, at most
        NEIGHBOURHOOD_MAX in all. Each option names the option it sits under
        (``parent``), always one listed before it, so they form a tree."""
        parent_of: Dict[str, Optional[str]] = {}

        def add(term: str, parent: Optional[str]) -> None:
            if term not in parent_of and len(parent_of) < NEIGHBOURHOOD_MAX:
                parent_of[term] = parent

        for term in suggested:
            above: Optional[str] = None
            for step in self.path_to_top(term):
                add(step, above)
                above = step
        for term in suggested:
            parent = parent_of.get(term)
            if parent is not None:
                for sibling in self.children(parent)[:_SIBLINGS_MAX]:
                    add(sibling, parent)
            for child in self.children(term, parts=True)[:_CHILDREN_MAX]:
                add(child, term)

        for term in suggested:  # never cut, even past the limit
            parent_of.setdefault(term, None)
        options = []
        for term, parent in parent_of.items():
            if parent not in parent_of:
                parent = None  # its path was cut: shown at the top
            option: Dict[str, Any] = {"id": term, "label": self.label(term)}
            if parent is not None:
                option["parent"] = parent
            notes = [SUGGESTED] if term in suggested else []
            if self.definition(term):
                notes.append(self.definition(term))
            if notes:
                option["details"] = " · ".join(notes)
            options.append(option)
        return options


_EXACT_SYNONYM = re.compile(r'^"(.+?)"\s+EXACT\b')
_QUOTED = re.compile(r'^"(.+?)"')


def _key(name: str) -> str:
    return " ".join(name.split()).casefold()


def _obsolete(data: Dict[str, Any]) -> bool:
    return str(data.get("is_obsolete", "")).lower() == "true"


_LOADED: Dict[str, Ontology] = {}


def ontology_names() -> List[str]:
    """The ontologies that can be loaded (lobster-vector's OBO_URLS)."""
    return sorted(_ontology_urls())


def _ontology_urls() -> Dict[str, str]:
    try:
        from lobster.vector.ontology_graph import OBO_URLS
    except ImportError:
        raise OntologyUnavailable(
            "ontology browsing isn't installed (the lobster-vector package)"
        ) from None
    return OBO_URLS


def load_ontology(name: str) -> Ontology:
    """The ontology called ``name``, loaded once per process.

    Raises ValueError for an ontology Lobster doesn't know, and
    OntologyUnavailable when it can't be loaded here.
    """
    if name in _LOADED:
        return _LOADED[name]
    urls = _ontology_urls()
    if name not in urls:
        raise ValueError(
            f"unknown ontology {name!r}; available: {', '.join(sorted(urls))}"
        )
    try:
        import obonet
    except ImportError:
        raise OntologyUnavailable(
            "ontology browsing isn't installed (the obonet package)"
        ) from None
    path = _downloaded(name, urls[name])
    try:
        graph = obonet.read_obo(str(path))
    except Exception as error:  # a broken file: download it again next time
        path.unlink(missing_ok=True)
        raise OntologyUnavailable(
            f"couldn't read the {name} ontology ({type(error).__name__})"
        ) from error
    logger.info("Loaded ontology %s: %d terms", name, graph.number_of_nodes())
    _LOADED[name] = Ontology(name, graph)
    return _LOADED[name]


def _downloaded(name: str, url: str) -> Path:
    """The ontology's file in the cache, downloaded first if it isn't there."""
    path = ONTOLOGY_CACHE / f"{name}.obo"
    if path.is_file():
        return path
    logger.info("Downloading ontology %s from %s", name, url)
    try:
        ONTOLOGY_CACHE.mkdir(parents=True, exist_ok=True)
        handle, partial = tempfile.mkstemp(dir=ONTOLOGY_CACHE, suffix=".part")
        with (
            os.fdopen(handle, "wb") as out,
            urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as response,
        ):
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        os.replace(partial, path)  # whole files only, never half a download
    except OSError as error:
        if "partial" in locals():
            Path(partial).unlink(missing_ok=True)
        raise OntologyUnavailable(
            f"couldn't download the {name} ontology ({type(error).__name__})"
        ) from error
    return path


# ---------------------------------------------------------------------------
# Building a question
# ---------------------------------------------------------------------------


def expand_view_sources(
    context: Any,
    workspace: Optional[Union[str, Path]],
    load: Optional[Callable[[str], Ontology]] = None,
) -> Tuple[Any, List[Problem], List[str]]:
    """Fill each top-level field's ``options_from`` and ``sequence_from``.

    Returns the new context, the problems that stop it, and notes for the
    supervisor (an ontology that couldn't be loaded). A context with neither
    is returned unchanged. ``load`` replaces load_ontology (tests).
    """
    fields = context.get("fields") if isinstance(context, dict) else None
    if not isinstance(fields, list) or not any(
        isinstance(field, dict)
        and ("options_from" in field or "sequence_from" in field)
        for field in fields
    ):
        return context, [], []
    load = load or load_ontology
    problems: List[Problem] = []
    notes: List[str] = []
    expanded: List[Any] = []
    files = list(context.get("data_files") or [])
    for index, field in enumerate(fields):
        where = f"fields.{index}"
        if isinstance(field, dict) and "options_from" in field:
            field, found, note = _expand_ontology(field, f"{where}.options_from", load)
            problems += found
            notes += [note] if note else []
        if isinstance(field, dict) and "sequence_from" in field:
            field, found = _expand_sequence(field, f"{where}.sequence_from", workspace)
            problems += found
            file = field["sequence_from"].get("file") if not found else None
            if file and file not in files:
                files.append(file)
        expanded.append(field)
    if problems:
        return context, problems, []
    return {**context, "fields": expanded, "data_files": files}, [], notes


def _spec_problems(error: ValidationError, where: str) -> List[Problem]:
    return [
        _problem(
            ".".join([where, *(str(part) for part in item["loc"])]),
            item["msg"].removeprefix("Value error, "),
        )
        for item in error.errors()
    ]


def _expand_ontology(
    field: Dict[str, Any], where: str, load: Callable[[str], Ontology]
) -> Tuple[Dict[str, Any], List[Problem], Optional[str]]:
    try:
        spec = OptionsFrom.model_validate(field["options_from"])
    except ValidationError as error:
        return field, _spec_problems(error, where), None
    if field.get("type") != "choice":
        return field, [_problem(where, "options_from is only for a choice field")], None
    if field.get("options"):
        return (
            field,
            [
                _problem(
                    where,
                    "options_from fills the options from the ontology; leave "
                    "options out",
                )
            ],
            None,
        )
    try:
        ontology = load(spec.ontology)
    except ValueError as error:
        return field, [_problem(f"{where}.ontology", str(error))], None
    except OntologyUnavailable as why:
        return _plain_choice(field, spec, str(why)), [], _unavailable_note(spec, why)

    found = {name: ontology.find(name) for name in spec.suggest}
    unknown = [name for name, term in found.items() if term is None]
    if unknown:
        return (
            field,
            [
                _problem(
                    f"{where}.suggest",
                    f"no term named {', '.join(repr(n) for n in unknown)} in "
                    f"{spec.ontology}; use a term's exact name or id",
                )
            ],
            None,
        )
    suggested = list(dict.fromkeys(found.values()))
    built = {
        **field,
        "options": ontology.neighbourhood(suggested),
        "options_from": spec.model_dump(),
    }
    if field.get("default") is not None:
        default, problems = _default_as_ids(field, ontology, f"{where}.default")
        if problems:
            return field, problems, None
        built["default"] = default
    return built, [], None


def _default_as_ids(
    field: Dict[str, Any], ontology: Ontology, where: str
) -> Tuple[Any, List[Problem]]:
    """The field's default with each term given by name turned into its id."""
    problems: List[Problem] = []

    def one(value: Any) -> Any:
        name = value.get("id") if isinstance(value, dict) else value
        if not isinstance(name, str):
            return value  # the field's own check names what is wrong
        term = ontology.find(name)
        if term is None:
            problems.append(_problem(where, f"no term named {name!r}"))
            return value
        return {"id": term}

    default = field["default"]
    if field.get("for_each") and isinstance(default, dict):
        return {item: one(value) for item, value in default.items()}, problems
    return one(default), problems


def _plain_choice(field: Dict[str, Any], spec: OptionsFrom, why: str) -> Dict[str, Any]:
    """The field as a plain choice of the suggested names, with a notice
    for the user: the ontology can't be shown here."""
    notice = (
        f"Lobster: {why}, so only the suggested terms are shown. Install "
        f"{INSTALL_HINT} to browse the {spec.ontology} ontology."
    )
    built = {
        key: value
        for key, value in field.items()
        if key not in ("options_from", "view")
    }
    names = list(dict.fromkeys(spec.suggest))
    built["options"] = [{"id": name} for name in names]
    built["details"] = "\n".join(
        part for part in (notice, field.get("details")) if part
    )
    # A default can only pre-fill a name that is now an option.
    if field.get("default") is not None:

        def shown(value: Any) -> bool:
            name = value.get("id") if isinstance(value, dict) else value
            return name in names

        default = field["default"]
        if field.get("for_each") and isinstance(default, dict):
            kept = {item: value for item, value in default.items() if shown(value)}
            built["default"] = kept or None
        elif not shown(default):
            built["default"] = None
    return built


def _unavailable_note(spec: OptionsFrom, why: Exception) -> str:
    return (
        f"The ontology view isn't available here: {why}. The question was "
        "asked as a plain choice of your suggested names, so answers are those "
        f"names, not {spec.ontology} ids. The user was told how to install it. "
        "Don't ask with the ontology view again in this session."
    )


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------


def _expand_sequence(
    field: Dict[str, Any], where: str, workspace: Optional[Union[str, Path]]
) -> Tuple[Dict[str, Any], List[Problem]]:
    try:
        spec = SequenceFrom.model_validate(field["sequence_from"])
    except ValidationError as error:
        return field, _spec_problems(error, where)
    if field.get("type") != "choice":
        return field, [_problem(where, "sequence_from is only for a choice field")]
    if spec.sequence is not None or spec.kind is not None:
        return field, [
            _problem(where, "leave sequence and kind out; they are read from the file")
        ]
    if workspace is None:
        return field, [
            _problem(where, "no workspace is open, so the file can't be read")
        ]
    try:
        record, letters, kind = read_fasta(workspace, spec.file, spec.id)
    except ValueError as error:
        return field, [_problem(where, str(error))]
    built = dict(field)
    built["sequence_from"] = {
        "file": spec.file,
        "id": record,
        "sequence": letters,
        "kind": kind,
    }
    return built, []


_NOT_LETTERS = re.compile(r"[^A-Za-z*]")


def read_fasta(
    workspace: Union[str, Path], raw: str, record: Optional[str]
) -> Tuple[str, str, str]:
    """(record id, letters, kind) of one record of a workspace FASTA file.

    ``record`` may be left out when the file holds one record. Raises
    ValueError saying what is wrong.
    """
    path = _file_inside(Path(workspace).resolve(), Path(raw), raw)
    try:
        if path.stat().st_size > _FASTA_BYTES_MAX:
            raise ValueError(f"{raw!r} is larger than {_FASTA_BYTES_MAX >> 20} MB")
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"can't read {raw!r} ({type(error).__name__})") from error
    records: Dict[str, List[str]] = {}
    current: Optional[List[str]] = None
    for line in text.splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0] if line[1:].split() else ""
            if not name:
                raise ValueError(f"{raw!r} has a record with no id")
            if name in records:
                raise ValueError(f"{raw!r} has the record {name!r} twice")
            current = records[name] = []
        elif line.strip() and not line.startswith(";"):
            if current is None:
                raise ValueError(f"{raw!r} is not a FASTA file (no '>' header first)")
            current.append(line)
    if not records:
        raise ValueError(f"{raw!r} has no FASTA records")
    if record is None:
        if len(records) > 1:
            raise ValueError(
                f"{raw!r} has {len(records)} records "
                f"({_some(list(records))}); say which in sequence_from.id"
            )
        record = next(iter(records))
    if record not in records:
        raise ValueError(
            f"{raw!r} has no record {record!r}; its records are "
            f"{_some(list(records))}"
        )
    letters = _NOT_LETTERS.sub("", "".join(records[record])).upper()
    if not letters:
        raise ValueError(f"record {record!r} in {raw!r} is empty")
    if len(letters) > SEQUENCE_MAX_LENGTH:
        raise ValueError(
            f"record {record!r} has {len(letters)} letters; one question shows "
            f"at most {SEQUENCE_MAX_LENGTH}"
        )
    return record, letters, _kind(letters)


def _kind(letters: str) -> str:
    used = set(letters)
    if used <= set("ACGTN"):
        return "dna"
    if used <= set("ACGUN"):
        return "rna"
    return "protein"


def reread_sequences(
    request: DecisionRequest, workspace: Optional[Union[str, Path]]
) -> DecisionRequest:
    """After a stale answer, read every sequence a field shows again, so the
    re-ask shows the file as it is now. A sequence that can't be read again,
    or no longer fits the regions, keeps the old one, and a problem says why
    (its file stays changed, so an answer stays stale until it is fixed)."""
    if workspace is None or not any(
        getattr(field, "sequence_from", None) for field in request.fields
    ):
        return request
    fields: List[Any] = []
    problems = list(request.problems)
    for field in request.fields:
        spec = getattr(field, "sequence_from", None)
        if spec is None:
            fields.append(field)
            continue
        try:
            _, letters, kind = read_fasta(workspace, spec.file, spec.id)
            rebuilt = type(field).model_validate(
                {
                    **field.model_dump(exclude_none=True),
                    "sequence_from": {
                        **spec.model_dump(exclude_none=True),
                        "sequence": letters,
                        "kind": kind,
                    },
                }
            )
        except ValueError as error:
            found = (
                problems_of(error, field.key)
                if isinstance(error, ValidationError)
                else [_problem(field.key, str(error))]
            )
            problems += [
                _problem(field.key, f"couldn't read {spec.file} again: {p.message}")
                for p in found
            ]
            fields.append(field)
            continue
        if letters != spec.sequence:
            problems.append(_problem(field.key, f"the sequence in {spec.file} changed"))
        fields.append(rebuilt)
    return request.model_copy(update={"fields": fields, "problems": problems})
