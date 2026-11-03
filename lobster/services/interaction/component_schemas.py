"""HITL decision contract: the request and answer shapes for ``ask_user``.

A request is an envelope (identity, versions) around a letter:
a question and a list of fields. Every field is built from four types
(choice, number, text, record) and two repeaters (``count``, ``for_each``);
a choice option may carry its own fields. Views say how a field is drawn and
never change the answer. Profiles add a named domain meaning and its checks.
Checks are named rules a question carries (``context.checks``), composed from
the ``CHECKS`` registry.

The answer mirrors the request. ``check_answer`` validates an answer against
its request: identity and versions first, then the data files, the shape and
profiles, then the answer-time checks. ``reask`` builds the next version of a
request after a refused answer, keeping what the user got right.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "lobster.ask/1"

# What a request may offer. "free_text" is never offered: it is how an older
# terminal, which could only show the question as text, sends the user's reply.
Action = Literal["submit", "defer", "cancel"]
AnswerAction = Literal["submit", "defer", "cancel", "free_text"]

# A check as the supervisor writes it: {"check": <name in CHECKS>, ...params}.
CheckSpec = Dict[str, Any]


class Problem(BaseModel):
    """One reason an answer or a question was refused.

    ``path`` names where: a field key ("groups"), a row ("qc.S02"), an
    envelope key ("responds_to.request_version") or "data_files".
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}" if self.path else self.message


def _problem(path: str, message: str) -> Problem:
    return Problem(path=path, message=message)


# ---------------------------------------------------------------------------
# Items: options and for_each entries share one shape
# ---------------------------------------------------------------------------


class Item(BaseModel):
    """One option or one for_each entry.

    ``id`` is the stable identity the answer refers to (a sample ID, an
    accession). ``label`` is for display only and may repeat.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Stable identity, never a row number")
    label: str = Field(default="", description="Display text; may repeat")
    details: Optional[str] = Field(
        default=None, description="Evidence shown beside the item"
    )
    unavailable: Optional[str] = Field(
        default=None,
        description="Reason this item can't be picked or answered; shown greyed out",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_plain_string(cls, value: Any) -> Any:
        """Allow "S01" as shorthand for {"id": "S01"}."""
        if isinstance(value, str):
            return {"id": value}
        return value

    @model_validator(mode="after")
    def _label_defaults_to_id(self) -> "Item":
        if not self.label:
            self.label = self.id
        return self


class Option(Item):
    """A choice option. ``fields`` appear only when this option is picked."""

    fields: Optional[List["QuestionField"]] = Field(
        default=None,
        min_length=1,
        description="Extra fields shown only when this option is picked",
    )

    @model_validator(mode="after")
    def _check_option_fields(self) -> "Option":
        if self.fields:
            _check_siblings(self.fields)
        return self


class Count(BaseModel):
    """How many values a field takes. The answer becomes a list."""

    model_config = ConfigDict(extra="forbid")

    min: int = Field(default=1, ge=0, description="Fewest values; 0 = optional")
    max: Optional[int] = Field(
        default=None, ge=1, description="Most values; left out = no limit"
    )

    @model_validator(mode="after")
    def _check_limits(self) -> "Count":
        if self.max is not None and self.min > self.max:
            raise ValueError(
                f"count.min ({self.min}) is larger than count.max ({self.max})"
            )
        return self


# ---------------------------------------------------------------------------
# Registries: views and profiles
# ---------------------------------------------------------------------------

# Richer displays a field may ask for. "applies_to" is the field type, or
# "fields" for the question as a whole; "needs" lists keys that must be set.
# A surface without the view draws the plain controls; the answer is the same.
# The ask_user docstring is generated from this dict.
VIEWS: Dict[str, Dict[str, Any]] = {
    "slider": {
        "applies_to": "number",
        "needs": ["min", "max"],
        "use_when": "one number between a known min and max, e.g. a cutoff",
        "example": {
            "key": "mito_cutoff",
            "type": "number",
            "view": "slider",
            "min": 0,
            "max": 0.5,
            "default": 0.2,
        },
    },
    "plate_map": {
        "applies_to": "choice",
        "needs": ["for_each"],
        "use_when": "the same choice for each sample, shown as wells on a plate",
        "example": {
            "key": "condition",
            "type": "choice",
            "view": "plate_map",
            "for_each": ["S01", "S02", "S03"],
            "options": ["control", "treated"],
        },
    },
    "dashboard": {
        "applies_to": "fields",
        "needs": [],
        "use_when": "several linked decisions on one screen; call "
        "get_dashboard_schema first",
        "example": None,
        "check": lambda owner: _check_dashboard(owner),
    },
}


def _check_interval(value: Any) -> None:
    if value["from"] > value["to"]:
        raise ValueError(f"from ({value['from']}) is larger than to ({value['to']})")


# Named domain meanings. "applies_to" is the field type, "fields" the required
# sub-fields; "check" receives the cleaned answer and raises ValueError.
PROFILES: Dict[str, Dict[str, Any]] = {
    "interval": {
        "applies_to": "record",
        "fields": {"from": "number", "to": "number"},
        "use_when": "a from/to pair; checks from <= to",
        "check": _check_interval,
    },
}


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


class _FieldBase(BaseModel):
    """Keys every field has."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, description="Name the answer comes back under")
    question: Optional[str] = Field(
        default=None,
        description="What to ask; may be left out when there is only one field",
    )
    details: Optional[str] = Field(
        default=None, description="Evidence or a recommendation; shown, never applied"
    )
    default: Any = Field(
        default=None, description="Pre-filled value, shaped exactly like the answer"
    )
    count: Optional[Count] = Field(
        default=None, description="How many values; left out = exactly one"
    )
    for_each: Optional[List[Item]] = Field(
        default=None,
        min_length=1,
        description="Ask this field once per item; the answer is keyed by item id",
    )
    view: str = Field(default="default", description='"default" or a name in VIEWS')
    profile: Optional[str] = Field(default=None, description="A name in PROFILES")
    depends_on: List[str] = Field(
        default_factory=list,
        description="Sibling keys this field depends on; changing them makes it stale",
    )

    @model_validator(mode="after")
    def _check_common(self) -> "_FieldBase":
        if self.for_each:
            _check_unique_ids(self.for_each, "for_each")
        _check_view(self.view, self.type, self)
        if self.profile is not None:
            _check_profile(self)
        if self.default is not None:
            try:
                check_value(self, self.default, f"{self.key}.default", partial=True)
            except AnswerRejected as error:
                raise ValueError(f"default is not a valid answer: {error}") from None
        return self


class ChoiceField(_FieldBase):
    """Pick options by id. Answer: {"id"}, plus "fields" for an option with fields."""

    type: Literal["choice"]
    options: List[Option] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_options(self) -> "ChoiceField":
        _check_unique_ids(self.options, "options")
        pickable = [option for option in self.options if not option.unavailable]
        if not pickable:
            raise ValueError("every option is unavailable; nothing can be picked")
        if self.count and self.count.min > len(pickable):
            raise ValueError(
                f"count.min ({self.count.min}) asks for more picks than there are "
                f"available options ({len(pickable)})"
            )
        return self


class NumberField(_FieldBase):
    """A number. ``min``/``max`` are value bounds, never "how many"."""

    type: Literal["number"]
    min: Optional[float] = Field(default=None, description="Lowest allowed value")
    max: Optional[float] = Field(default=None, description="Highest allowed value")
    integer: bool = Field(default=False, description="Whole numbers only")
    unit: Optional[str] = Field(default=None, description='e.g. "mmol/L", "h"')

    @model_validator(mode="before")
    @classmethod
    def _check_bounds(cls, data: Any) -> Any:
        # Before-mode, so it runs ahead of _check_common's check of `default`.
        if isinstance(data, dict):
            low, high = data.get("min"), data.get("max")
            if low is not None and high is not None and low > high:
                raise ValueError(f"min ({low}) is larger than max ({high})")
        return data


class TextField(_FieldBase):
    """Free text, only when the answer can't be listed or counted."""

    type: Literal["text"]


class RecordField(_FieldBase):
    """Several fields answered together. Answer: {sub-key: answer}."""

    type: Literal["record"]
    fields: List["QuestionField"] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_record(self) -> "RecordField":
        _check_siblings(self.fields)
        return self


QuestionField = Annotated[
    Union[ChoiceField, NumberField, TextField, RecordField],
    Field(discriminator="type"),
]

Option.model_rebuild()
RecordField.model_rebuild()


class AskUserContext(BaseModel):
    """What the supervisor passes as ``ask_user``'s ``context``."""

    model_config = ConfigDict(extra="forbid")

    view: str = Field(
        default="default", description='"default" or a view that applies to "fields"'
    )
    fields: List[QuestionField] = Field(min_length=1)
    checks: List[CheckSpec] = Field(
        default_factory=list, description="Named rules from CHECKS, with parameters"
    )
    data_files: List[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=list,
        description="Workspace-relative paths of the files the decision is about",
    )

    @model_validator(mode="after")
    def _check_fields(self) -> "AskUserContext":
        _check_siblings(self.fields)
        if len(self.fields) > 1:
            missing = [field.key for field in self.fields if not field.question]
            if missing:
                raise ValueError(
                    f"fields {missing} need a question; only a single field may "
                    "leave it out"
                )
        _check_view(self.view, "fields", self)
        # Shape only. Request-time checks run once, in ask_user (check_request).
        _check_check_specs(self.checks, self.fields)
        duplicates = sorted(
            {p for p in self.data_files if self.data_files.count(p) > 1}
        )
        if duplicates:
            raise ValueError(f"data_files has duplicate paths {duplicates}")
        return self


# ---------------------------------------------------------------------------
# Data files: fingerprints taken by Python, never supplied by the LLM
# ---------------------------------------------------------------------------


class FileFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="Workspace-relative path")
    sha256: Optional[str] = Field(
        default=None,
        description="Hex digest; None only in a fresh reading of a file that is gone",
    )


class FileSnapshot(BaseModel):
    """What the decision's data files looked like when the question was asked."""

    model_config = ConfigDict(extra="forbid")

    files: List[FileFingerprint] = Field(min_length=1)


_CHUNK = 1024 * 1024


def fingerprint(
    paths: Iterable[str], workspace: Union[str, Path], missing_ok: bool = False
) -> FileSnapshot:
    """sha256 of each workspace file, read in chunks.

    Paths must be relative and stay inside the workspace (no absolute paths,
    no "..", no links out). A missing or unreadable file raises ValueError,
    or with ``missing_ok`` gets ``sha256 = None`` (used when re-reading files
    at answer time, where a gone file means the data changed).
    """
    root = Path(workspace).resolve()
    files = []
    for raw in paths:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"{raw!r} must be a path inside the workspace "
                "(relative, without '..')"
            )
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"{raw!r} leads outside the workspace")
        try:
            digest = _sha256(resolved)
        except OSError as error:
            if not missing_ok:
                reason = error.strerror or type(error).__name__
                raise ValueError(f"can't read {raw!r} in the workspace ({reason})")
            digest = None
        files.append(FileFingerprint(path=relative.as_posix(), sha256=digest))
    return FileSnapshot(files=files)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


class DecisionRequest(BaseModel):
    """The envelope ask_user sends to a surface. Built by code, never by the LLM.

    ``prefilled_from_answer`` says where the defaults came from: True when a
    re-ask filled them from the user's submitted answer, False when they are
    still the supervisor's proposals (a first ask, or a re-ask after a
    misrouted or malformed reply to one). A surface may show defaults as the
    user's answer only when it is True.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["lobster.ask/1"] = Field(default=SCHEMA_VERSION, alias="schema")
    decision_id: str = Field(default_factory=lambda: f"dec-{uuid.uuid4()}")
    request_version: int = Field(default=1, ge=1)
    actions: List[Action] = Field(default_factory=lambda: ["submit", "defer", "cancel"])
    question: str = Field(min_length=1)
    view: str = "default"
    fields: List[QuestionField] = Field(min_length=1)
    checks: List[CheckSpec] = Field(default_factory=list)
    problems: List[Problem] = Field(
        default_factory=list, description="Why the previous answer was refused"
    )
    snapshot: Optional[FileSnapshot] = Field(
        default=None, description="The data files' fingerprints, taken by ask_user"
    )
    prefilled_from_answer: bool = Field(
        default=False,
        description="True when the defaults are the user's submitted answer",
    )

    @model_validator(mode="after")
    def _single_field_uses_main_question(self) -> "DecisionRequest":
        """A lone field without a question shows the main question."""
        if len(self.fields) == 1 and not self.fields[0].question:
            self.fields[0].question = self.question
        _check_check_specs(self.checks, self.fields)
        return self

    def to_wire(self) -> Dict[str, Any]:
        """JSON-ready dict, as sent through interrupt() and the bridge."""
        return self.model_dump(by_alias=True, exclude_none=True)


class RespondsTo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(min_length=1)
    request_version: int = Field(ge=1)


class DecisionAnswer(BaseModel):
    """What a surface sends back for one DecisionRequest."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["lobster.ask/1"] = Field(default=SCHEMA_VERSION, alias="schema")
    responds_to: RespondsTo
    answer_id: str = Field(default_factory=lambda: f"ans-{uuid.uuid4()}")
    action: AnswerAction
    answer: Dict[str, Any] = Field(default_factory=dict)
    comment: Optional[str] = None
    text: Optional[str] = Field(
        default=None, description="The user's own words; only for free_text"
    )

    @model_validator(mode="after")
    def _text_only_for_free_text(self) -> "DecisionAnswer":
        if self.action == "free_text":
            if not self.text or not self.text.strip():
                raise ValueError("a free_text answer needs non-empty text")
            self.text = self.text.strip()
        elif self.text is not None:
            raise ValueError("text is only for the free_text action")
        return self


# ---------------------------------------------------------------------------
# Checking answers
# ---------------------------------------------------------------------------


class AnswerRejected(ValueError):
    """An answer that doesn't fit its request. ``problems`` lists every reason."""

    def __init__(self, problems: List[Problem]):
        self.problems = list(problems)
        super().__init__("; ".join(str(problem) for problem in self.problems))


class AnswerMisrouted(AnswerRejected):
    """An answer for another decision. Never consumed, never used to pre-fill."""


class AnswerStale(AnswerRejected):
    """An answer to an old question version, or to data that has changed since.

    ``snapshot`` is the fresh reading of the data files when they changed.
    """

    def __init__(
        self, problems: List[Problem], snapshot: Optional[FileSnapshot] = None
    ):
        super().__init__(problems)
        self.snapshot = snapshot


def check_answer(
    request: DecisionRequest,
    answer: DecisionAnswer,
    current_snapshot: Optional[FileSnapshot] = None,
) -> Dict[str, Any]:
    """Check a surface's answer against its request.

    Returns what the supervisor receives. Raises AnswerMisrouted for another
    decision, AnswerStale for another version or for changed data files
    (``current_snapshot``: the files read again now; the caller passes it
    whenever the request has a snapshot), AnswerRejected otherwise, listing
    every problem. A ``free_text`` answer is returned unchecked: it is the
    user's own words, not an answer shaped like the fields.
    """
    if answer.responds_to.decision_id != request.decision_id:
        raise AnswerMisrouted(
            [
                _problem(
                    "responds_to.decision_id",
                    f"this answer is for decision {answer.responds_to.decision_id}, "
                    f"not {request.decision_id}",
                )
            ]
        )
    if answer.responds_to.request_version != request.request_version:
        raise AnswerStale(
            [
                _problem(
                    "responds_to.request_version",
                    f"this answer is for version {answer.responds_to.request_version} "
                    f"of the question; the current version is "
                    f"{request.request_version}",
                )
            ]
        )
    result: Dict[str, Any] = {
        "decision_id": request.decision_id,
        "answer_id": answer.answer_id,
        "action": answer.action,
    }
    if answer.action == "free_text":
        return {**result, "text": answer.text}
    if answer.action not in request.actions:
        raise AnswerRejected(
            [_problem("action", f"action {answer.action!r} is not allowed here")]
        )
    if answer.comment:
        result["comment"] = answer.comment
    if answer.action == "submit":
        if request.snapshot is not None and current_snapshot is not None:
            changed = _changed_files(request.snapshot, current_snapshot)
            if changed:
                raise AnswerStale(
                    [
                        _problem(
                            "data_files",
                            "the data changed while you were reviewing it: " + path,
                        )
                        for path in changed
                    ],
                    snapshot=current_snapshot,
                )
        result["answer"] = _check_submitted(request, answer.answer)
    return result


def _changed_files(asked: FileSnapshot, now: FileSnapshot) -> List[str]:
    """Paths whose content differs from when the question was asked, or is gone."""
    current = {entry.path: entry.sha256 for entry in now.files}
    return [
        entry.path
        for entry in asked.files
        if current.get(entry.path) is None or current[entry.path] != entry.sha256
    ]


def _check_submitted(request: DecisionRequest, answer: Any) -> Dict[str, Any]:
    """Shape checks, then the answer-time checks; every problem is collected.

    A check whose fields failed their shape check is skipped: their problems
    are already listed, and a check on a broken value would only add noise.
    """
    cleaned, problems = _collect_fields(request.fields, answer, "")
    by_key = {field.key: field for field in request.fields}
    for spec in request.checks:
        entry = CHECKS[spec["check"]]
        if entry["on"] == "answer":
            problems += entry["run"](spec, by_key, cleaned)
    if problems:
        raise AnswerRejected(problems)
    return cleaned


def check_request(context: AskUserContext) -> List[Problem]:
    """Run the request-time checks on the supervisor's context.

    Called once, by ask_user, before anything is shown. Never inside a model
    validator, so a re-asked request (whose defaults are the user's own
    answers) can't trip them.
    """
    by_key = {field.key: field for field in context.fields}
    problems: List[Problem] = []
    for spec in context.checks:
        entry = CHECKS[spec["check"]]
        if entry["on"] == "request":
            problems += entry["run"](spec, by_key, None)
    return problems


def check_value(field: Any, value: Any, path: str, partial: bool = False) -> Any:
    """Check one field's answer (for_each, count, type, profile); return it cleaned.

    ``partial`` allows a for_each value to leave items out: a default may
    pre-fill only the rows it knows.
    """
    if not field.for_each:
        return _check_counted(field, value, path)
    if not isinstance(value, dict):
        raise AnswerRejected(
            [_problem(path, "expected an answer per item, keyed by id")]
        )
    open_ids = [item.id for item in field.for_each if not item.unavailable]
    problems = (
        []
        if partial
        else [_problem(f"{path}.{i}", "no answer") for i in open_ids if i not in value]
    )
    problems += [
        _problem(f"{path}.{i}", "not an item here") for i in value if i not in open_ids
    ]
    cleaned: Dict[str, Any] = {}
    for item_id in open_ids:
        if item_id not in value:
            continue
        try:
            cleaned[item_id] = _check_counted(
                field, value[item_id], f"{path}.{item_id}"
            )
        except AnswerRejected as error:
            problems += error.problems
    if problems:
        raise AnswerRejected(problems)
    return cleaned


def _check_counted(field: Any, value: Any, path: str) -> Any:
    """Apply ``count``: one value, or a list of values within the limits."""
    if field.count is None:
        return _check_single(field, value, path)
    if not isinstance(value, list):
        raise AnswerRejected([_problem(path, "expected a list")])
    low, high = field.count.min, field.count.max
    if len(value) < low or (high is not None and len(value) > high):
        limit = f"{low} or more" if high is None else f"between {low} and {high}"
        raise AnswerRejected([_problem(path, f"{len(value)} given; give {limit}")])
    cleaned, problems = [], []
    for index, entry in enumerate(value):
        try:
            cleaned.append(_check_single(field, entry, f"{path}[{index}]"))
        except AnswerRejected as error:
            problems += error.problems
    if field.type == "choice" and not problems:
        ids = [entry["id"] for entry in cleaned]
        if len(set(ids)) != len(ids):
            problems.append(_problem(path, "the same option was picked twice"))
    if problems:
        raise AnswerRejected(problems)
    return cleaned


def _check_single(field: Any, value: Any, path: str) -> Any:
    """Check one value against the field's type, then its profile."""
    cleaned = _TYPE_CHECKS[field.type](field, value, path)
    if field.profile:
        try:
            PROFILES[field.profile]["check"](cleaned)
        except ValueError as error:
            raise AnswerRejected([_problem(path, str(error))]) from None
    return cleaned


def _check_choice(field: ChoiceField, value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("id"), str):
        raise AnswerRejected([_problem(path, 'expected {"id": ...}')])
    extra = set(value) - {"id", "fields"}
    if extra:
        raise AnswerRejected([_problem(path, f"unexpected keys {sorted(extra)}")])
    option = next((o for o in field.options if o.id == value["id"]), None)
    if option is None:
        raise AnswerRejected([_problem(path, f"unknown option {value['id']!r}")])
    if option.unavailable:
        raise AnswerRejected(
            [
                _problem(
                    path, f"option {option.id!r} is unavailable ({option.unavailable})"
                )
            ]
        )
    if not option.fields:
        if value.get("fields"):
            raise AnswerRejected(
                [_problem(path, f"option {option.id!r} has no fields")]
            )
        return {"id": option.id}
    sub = value.get("fields")
    if not isinstance(sub, dict):
        raise AnswerRejected(
            [_problem(f"{path}.fields", f"option {option.id!r} needs its fields")]
        )
    return {
        "id": option.id,
        "fields": _check_fields(option.fields, sub, f"{path}.fields"),
    }


def _check_number(field: NumberField, value: Any, path: str) -> Union[int, float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnswerRejected([_problem(path, "expected a number")])
    if field.integer and not float(value).is_integer():
        raise AnswerRejected([_problem(path, f"{value} must be a whole number")])
    if field.min is not None and value < field.min:
        raise AnswerRejected([_problem(path, f"{value} is below min ({field.min})")])
    if field.max is not None and value > field.max:
        raise AnswerRejected([_problem(path, f"{value} is above max ({field.max})")])
    return int(value) if field.integer else value


def _check_text(field: TextField, value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnswerRejected([_problem(path, "expected non-empty text")])
    return value.strip()


def _check_record(field: RecordField, value: Any, path: str) -> Dict[str, Any]:
    return _check_fields(field.fields, value, path)


_TYPE_CHECKS: Dict[str, Callable[[Any, Any, str], Any]] = {
    "choice": _check_choice,
    "number": _check_number,
    "text": _check_text,
    "record": _check_record,
}


def _check_fields(fields: List[Any], answer: Any, path: str) -> Dict[str, Any]:
    """Check answers for a list of sibling fields; collect every problem."""
    cleaned, problems = _collect_fields(fields, answer, path)
    if problems:
        raise AnswerRejected(problems)
    return cleaned


def _collect_fields(
    fields: List[Any], answer: Any, path: str
) -> Tuple[Dict[str, Any], List[Problem]]:
    """Like _check_fields, but return the fields that passed with the problems."""
    prefix = f"{path}." if path else ""
    if not isinstance(answer, dict):
        return {}, [_problem(path or "answer", "expected an answer per field")]
    by_key = {field.key: field for field in fields}
    problems: List[Problem] = []
    for key, field in by_key.items():
        optional = field.count is not None and field.count.min == 0
        if key not in answer and not optional:
            problems.append(_problem(f"{prefix}{key}", "no answer"))
    problems += [
        _problem(f"{prefix}{key}", "not a field in this question")
        for key in answer
        if key not in by_key
    ]
    cleaned: Dict[str, Any] = {}
    for key, value in answer.items():
        if key not in by_key:
            continue
        try:
            cleaned[key] = check_value(by_key[key], value, f"{prefix}{key}")
        except AnswerRejected as error:
            problems += error.problems
    return cleaned, problems


# ---------------------------------------------------------------------------
# Asking again
# ---------------------------------------------------------------------------


def reask(
    request: DecisionRequest,
    problems: List[Problem],
    submitted: Optional[Dict[str, Any]],
    snapshot: Optional[FileSnapshot] = None,
) -> DecisionRequest:
    """The next version of ``request`` after a refused answer.

    Each field's default becomes the submitted value where that value is a
    valid default; for a for_each field every valid row is kept and only bad
    rows are dropped. Anything unusable keeps the previous default.
    ``submitted = None`` (an answer that was not for this question, or not
    an answer at all) changes no default, so ``prefilled_from_answer`` stays
    as it was; with a submitted answer it becomes True. ``snapshot``, a fresh
    reading of the data files, replaces the old fingerprints, except for a
    file that is gone, which keeps its old one so the answer stays stale
    until the file is back.
    """
    fields = request.fields
    prefilled = request.prefilled_from_answer
    if isinstance(submitted, dict):
        fields = [_refill(field, submitted) for field in request.fields]
        prefilled = True
    update: Dict[str, Any] = {
        "request_version": request.request_version + 1,
        "problems": list(problems),
        "fields": fields,
        "prefilled_from_answer": prefilled,
    }
    if snapshot is not None and request.snapshot is not None:
        fresh = {entry.path: entry.sha256 for entry in snapshot.files}
        update["snapshot"] = FileSnapshot(
            files=[
                FileFingerprint(
                    path=entry.path, sha256=fresh.get(entry.path) or entry.sha256
                )
                for entry in request.snapshot.files
            ]
        )
    return request.model_copy(update=update)


def _refill(field: Any, submitted: Dict[str, Any]) -> Any:
    """``field`` with the submitted value as its default, where it is valid."""
    if field.key not in submitted:
        return field
    value = submitted[field.key]
    if field.for_each:
        if not isinstance(value, dict):
            return field
        rows = dict(field.default) if isinstance(field.default, dict) else {}
        for item_id, row in value.items():
            try:
                rows.update(check_value(field, {item_id: row}, field.key, partial=True))
            except AnswerRejected:
                rows.pop(item_id, None)
        return field.model_copy(update={"default": rows or None})
    try:
        default = check_value(field, value, field.key, partial=True)
    except AnswerRejected:
        return field
    return field.model_copy(update={"default": default})


# ---------------------------------------------------------------------------
# Helpers for request validation
# ---------------------------------------------------------------------------


def _check_unique_ids(items: List[Item], where: str) -> None:
    ids = [item.id for item in items]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{where} have duplicate ids {duplicates}")


def _check_siblings(fields: List[Any]) -> None:
    """Sibling keys are unique; depends_on names existing siblings, without cycles."""
    keys = [field.key for field in fields]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ValueError(f"fields have duplicate keys {duplicates}")
    graph = {field.key: field.depends_on for field in fields}
    for key, deps in graph.items():
        for dep in deps:
            if dep == key:
                raise ValueError(f"{key} can't depend on itself")
            if dep not in graph:
                raise ValueError(
                    f"{key} depends on {dep!r}, which is not a sibling field"
                )
    visiting, done = set(), set()

    def visit(key: str) -> None:
        if key in done:
            return
        if key in visiting:
            raise ValueError(f"depends_on has a cycle through {key!r}")
        visiting.add(key)
        for dep in graph[key]:
            visit(dep)
        visiting.discard(key)
        done.add(key)

    for key in graph:
        visit(key)


def _check_view(view: str, field_type: str, owner: Any) -> None:
    """A view must be "default" or registered for this type, with what it needs."""
    if view == "default":
        return
    if view not in VIEWS:
        available = ", ".join(["default", *sorted(VIEWS)])
        raise ValueError(f"unknown view {view!r}; available views: {available}")
    entry = VIEWS[view]
    if entry["applies_to"] != field_type:
        raise ValueError(
            f"view {view!r} applies to {entry['applies_to']}, not {field_type}"
        )
    missing = [name for name in entry["needs"] if getattr(owner, name, None) is None]
    if missing:
        raise ValueError(f"view {view!r} needs {', '.join(missing)}")
    if "check" in entry:
        entry["check"](owner)


def _check_dashboard(owner: Any) -> None:
    """Each dashboard section must be a field the dashboard can draw as one panel."""
    if len(owner.fields) < 2:
        raise ValueError("view 'dashboard' needs at least 2 fields")
    for field in owner.fields:
        where = f"dashboard section {field.key!r}"
        if field.type == "record":
            raise ValueError(f"{where}: a record can't be a section; split it")
        if field.count and field.type != "choice":
            raise ValueError(f"{where}: count is only supported on a choice here")
        if field.for_each and field.type != "choice":
            raise ValueError(f"{where}: for_each is only supported on a choice here")
        if field.for_each and field.view != "plate_map":
            # A per-item list: one option per item, an option may ask for a reason.
            if field.count:
                raise ValueError(f"{where}: a per-item list takes one option per item")
            for option in field.options:
                if option.fields and not _is_one_required_text(option.fields):
                    raise ValueError(
                        f"{where}: option {option.id!r} may only have one required "
                        "text field (a reason) here"
                    )
        elif field.type == "choice" and any(o.fields for o in field.options):
            raise ValueError(f"{where}: options with fields aren't supported here")


def _is_one_required_text(fields: List[Any]) -> bool:
    """Exactly one text field, answered once (no count, no for_each)."""
    return (
        len(fields) == 1
        and fields[0].type == "text"
        and fields[0].count is None
        and not fields[0].for_each
    )


def _check_profile(field: Any) -> None:
    entry = PROFILES.get(field.profile)
    if entry is None:
        available = ", ".join(sorted(PROFILES)) or "none"
        raise ValueError(f"unknown profile {field.profile!r}; available: {available}")
    if entry["applies_to"] != field.type:
        raise ValueError(
            f"profile {field.profile!r} applies to {entry['applies_to']}, "
            f"not {field.type}"
        )
    shape = {sub.key: sub.type for sub in getattr(field, "fields", [])}
    if "fields" in entry and shape != entry["fields"]:
        raise ValueError(
            f"profile {field.profile!r} needs sub-fields {entry['fields']}, got {shape}"
        )


# ---------------------------------------------------------------------------
# Checks: named rules a question carries
# ---------------------------------------------------------------------------
#
# Each entry: "use_when" (one docstring line), "on" ("request": run once by
# ask_user on the supervisor's context, a failure is invalid_context; "answer":
# run by check_answer after the shape checks), "validate(spec, fields)" (raises
# ValueError when the spec doesn't fit the question) and
# "run(spec, fields, answer) -> List[Problem]". ``fields`` maps the top-level
# keys to fields; checks name top-level fields only. A check sees one answer
# (one question or one dashboard), never earlier ask_user calls.


def _params(
    spec: CheckSpec, required: Iterable[str], optional: Iterable[str] = ()
) -> None:
    name = spec["check"]
    allowed = {"check", *required, *optional}
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValueError(
            f"check {name!r}: unknown parameters {unknown}; "
            f"it takes {sorted(allowed - {'check'})}"
        )
    missing = [key for key in required if spec.get(key) is None]
    if missing:
        raise ValueError(f"check {name!r} needs {', '.join(missing)}")


def _ref(spec: CheckSpec, fields: Dict[str, Any], key: str, where: str) -> Any:
    if not isinstance(key, str) or key not in fields:
        raise ValueError(
            f"check {spec['check']!r}: no field {key!r} at the top level "
            f"(fields: {', '.join(fields)})"
        )
    return fields[key]


def _ids(spec: CheckSpec, value: Any, known: List[str], what: str, where: str):
    name = spec["check"]
    if not isinstance(value, list) or not all(isinstance(i, str) for i in value):
        raise ValueError(f"check {name!r}: {where} must be a list of {what} ids")
    if not value:
        raise ValueError(f"check {name!r}: {where} needs at least one {what}")
    for item_id in value:
        if item_id not in known:
            raise ValueError(
                f"check {name!r}: unknown {what} {item_id!r} in {where} "
                f"(known: {', '.join(known)})"
            )
    return value


def _per_item_choice(spec: CheckSpec, field: Any, where: str) -> None:
    if field.type != "choice" or not field.for_each:
        raise ValueError(
            f"check {spec['check']!r}: {where} {field.key!r} must be a per-item "
            "choice (a choice with for_each)"
        )


def _picked(value: Any) -> List[str]:
    """Option ids in one cleaned choice answer (one pick or a counted list)."""
    picks = value if isinstance(value, list) else [value]
    return [pick["id"] for pick in picks if isinstance(pick, dict)]


def _option_name(field: Any, option_id: str) -> str:
    option = next(o for o in field.options if o.id == option_id)
    if option.label != option.id:
        return f"'{option.label}' ({option.id})"
    return f"'{option.id}'"


def _item_names(field: Any, item_ids: List[str]) -> str:
    labels = {item.id: item.label for item in field.for_each or []}
    return ", ".join(
        (
            item_id
            if labels.get(item_id, item_id) == item_id
            else f"{item_id} ({labels[item_id]})"
        )
        for item_id in item_ids
    )


def _validate_each_option_kept(spec: CheckSpec, fields: Dict[str, Any]) -> None:
    _params(spec, ["field", "options"], ["minus"])
    field = _ref(spec, fields, spec["field"], "field")
    _per_item_choice(spec, field, "field")
    _ids(spec, spec["options"], [o.id for o in field.options], "option", "options")
    minus = spec.get("minus")
    if minus is None:
        return
    if isinstance(minus, dict) and set(minus) - {"field", "options"}:
        raise ValueError(
            "check 'each_option_kept': unknown parameters "
            f"{sorted(set(minus) - {'field', 'options'})} in minus; "
            'minus needs {"field", "options"}'
        )
    if not isinstance(minus, dict) or "field" not in minus or "options" not in minus:
        raise ValueError('check \'each_option_kept\': minus needs {"field", "options"}')
    other = _ref(spec, fields, minus["field"], "minus.field")
    if other.key == field.key:
        raise ValueError("check 'each_option_kept': minus.field must be another field")
    _per_item_choice(spec, other, "minus.field")
    _ids(
        spec, minus["options"], [o.id for o in other.options], "option", "minus.options"
    )


def _run_each_option_kept(
    spec: CheckSpec, fields: Dict[str, Any], answer: Optional[Dict[str, Any]]
) -> List[Problem]:
    """Serves the rule that no comparison group may end up empty, also after
    the items taken out by another decision (e.g. QC) are removed."""
    field, minus = fields[spec["field"]], spec.get("minus")
    answer = answer or {}
    if field.key not in answer or (minus and minus["field"] not in answer):
        return []  # a referenced field failed its shape check: already reported
    removed: Dict[str, List[str]] = {}
    if minus:
        for item_id, value in answer[minus["field"]].items():
            hits = [o for o in minus["options"] if o in _picked(value)]
            if hits:
                removed[item_id] = hits
    problems: List[Problem] = []
    for option_id in spec["options"]:
        members = [i for i, v in answer[field.key].items() if option_id in _picked(v)]
        kept = [i for i in members if i not in removed]
        name = _option_name(field, option_id)
        if not members:
            problems.append(
                _problem(field.key, f"{name} has no items; give it at least one")
            )
        elif not kept:
            used = [
                o for o in minus["options"] if any(o in removed[i] for i in members)
            ]
            message = (
                f"{name} has no items left after removing "
                f"{_item_names(field, members)} ({minus['field']}: {', '.join(used)})"
            )
            problems += [
                _problem(field.key, message),
                _problem(minus["field"], message),
            ]
    return problems


def _validate_reason_required(spec: CheckSpec, fields: Dict[str, Any]) -> None:
    _params(spec, ["field", "options"])
    field = _ref(spec, fields, spec["field"], "field")
    if field.type != "choice":
        raise ValueError(
            f"check 'reason_required': field {field.key!r} must be a choice"
        )
    _ids(spec, spec["options"], [o.id for o in field.options], "option", "options")


def _run_reason_required(
    spec: CheckSpec, fields: Dict[str, Any], answer: Optional[Dict[str, Any]]
) -> List[Problem]:
    """Serves the rule that taking an item out needs a stated reason: the
    listed options must ask for one, so the answer can't come back without it."""
    field = fields[spec["field"]]
    problems = []
    for option in field.options:
        if option.id not in spec["options"]:
            continue
        if not any(
            sub.type == "text" and sub.count is None and not sub.for_each
            for sub in option.fields or []
        ):
            problems.append(
                _problem(
                    field.key,
                    f"option {option.id!r} needs a required text field for the "
                    'reason, e.g. "fields": [{"key": "reason", "type": "text"}]',
                )
            )
    return problems


def _validate_no_default_for(spec: CheckSpec, fields: Dict[str, Any]) -> None:
    _params(spec, ["field"], ["items", "options"])
    field = _ref(spec, fields, spec["field"], "field")
    if spec.get("items") is not None:
        if not field.for_each:
            raise ValueError(
                f"check 'no_default_for': items need field {field.key!r} "
                "to have for_each"
            )
        _ids(spec, spec["items"], [i.id for i in field.for_each], "item", "items")
    if spec.get("options") is not None:
        if field.type != "choice":
            raise ValueError(
                f"check 'no_default_for': options need field {field.key!r} "
                "to be a choice"
            )
        _ids(spec, spec["options"], [o.id for o in field.options], "option", "options")


def _run_no_default_for(
    spec: CheckSpec, fields: Dict[str, Any], answer: Optional[Dict[str, Any]]
) -> List[Problem]:
    """Serves the rule that a suggestion or a warning never decides by itself:
    flagged or uncertain items are left for the user, not pre-filled."""
    field = fields[spec["field"]]
    items, options = spec.get("items"), spec.get("options")
    if field.default is None:
        return []
    after = "; leave it for the user to decide"
    if not field.for_each:
        picked = _picked(field.default) if field.type == "choice" else []
        if options is None:
            return [_problem(field.key, "this field may not have a default" + after)]
        hits = [o for o in options if o in picked]
        if not hits:
            return []
        names = ", ".join(_option_name(field, o) for o in hits)
        return [_problem(field.key, f"the default pre-selects {names}{after}")]
    rows = [i for i in field.default if items is None or i in items]
    if options is None:
        if not rows:
            return []
        message = f"the default pre-fills {_item_names(field, rows)}{after}"
        return [_problem(field.key, message)]
    hits = {o: [i for i in rows if o in _picked(field.default[i])] for o in options}
    return [
        _problem(
            field.key,
            f"the default pre-fills {_item_names(field, ids)} with "
            f"{_option_name(field, option_id)}{after}",
        )
        for option_id, ids in hits.items()
        if ids
    ]


CHECKS: Dict[str, Dict[str, Any]] = {
    "each_option_kept": {
        "on": "answer",
        "use_when": "field, options, optional minus {field, options}: each listed "
        "option of a per-item choice keeps at least one item; items given a minus "
        "option in the other per-item field are taken out first",
        "validate": _validate_each_option_kept,
        "run": _run_each_option_kept,
    },
    "reason_required": {
        "on": "request",
        "use_when": "field, options: each listed option must carry a required text "
        "field (the reason)",
        "validate": _validate_reason_required,
        "run": _run_reason_required,
    },
    "no_default_for": {
        "on": "request",
        "use_when": "field, optional items, optional options: the default may not "
        "pre-fill these items (with these options); the user decides them",
        "validate": _validate_no_default_for,
        "run": _run_no_default_for,
    },
}


def _check_check_specs(checks: List[CheckSpec], fields: List[Any]) -> None:
    """Each check names a registered check and fits the question (shape only)."""
    by_key = {field.key: field for field in fields}
    for spec in checks:
        if not isinstance(spec.get("check"), str):
            raise ValueError(
                'every check needs a "check" name, e.g. "each_option_kept"'
            )
        if spec["check"] not in CHECKS:
            available = ", ".join(sorted(CHECKS))
            raise ValueError(
                f"unknown check {spec['check']!r}; available checks: {available}"
            )
        CHECKS[spec["check"]]["validate"](spec, by_key)
