"""HITL decision contract: the request and answer shapes for ``ask_user``.

A request is an envelope (identity, versions) around a letter:
a question and a list of fields. Every field is built from four types
(choice, number, text, record) and two repeaters (``count``, ``for_each``);
a choice option may carry its own fields. Views say how a field is drawn and
never change the answer. Profiles add a named domain meaning and its checks.

The answer mirrors the request. ``check_answer`` validates an answer against
its request: identity and versions first, then the shape and profiles.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "lobster.ask/1"

Action = Literal["submit", "defer", "cancel"]


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
                raise ValueError(
                    "default is not a valid answer: " + "; ".join(error.problems)
                ) from None
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
        return self


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


class DecisionRequest(BaseModel):
    """The envelope ask_user sends to a surface. Built by code, never by the LLM."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["lobster.ask/1"] = Field(default=SCHEMA_VERSION, alias="schema")
    decision_id: str = Field(default_factory=lambda: f"dec-{uuid.uuid4()}")
    request_version: int = Field(default=1, ge=1)
    actions: List[Action] = Field(default_factory=lambda: ["submit", "defer", "cancel"])
    question: str = Field(min_length=1)
    view: str = "default"
    fields: List[QuestionField] = Field(min_length=1)

    @model_validator(mode="after")
    def _single_field_uses_main_question(self) -> "DecisionRequest":
        """A lone field without a question shows the main question."""
        if len(self.fields) == 1 and not self.fields[0].question:
            self.fields[0].question = self.question
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
    action: Action
    answer: Dict[str, Any] = Field(default_factory=dict)
    comment: Optional[str] = None


# ---------------------------------------------------------------------------
# Checking answers
# ---------------------------------------------------------------------------


class AnswerRejected(ValueError):
    """An answer that doesn't fit its request. ``problems`` lists every reason."""

    def __init__(self, problems: List[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


class AnswerStale(AnswerRejected):
    """An answer to an old question version."""


def check_answer(request: DecisionRequest, answer: DecisionAnswer) -> Dict[str, Any]:
    """Check a surface's answer against its request.

    Returns what the supervisor receives. Raises AnswerStale when the question
    changed, AnswerRejected (listing every problem) otherwise.
    """
    if answer.responds_to.decision_id != request.decision_id:
        raise AnswerRejected(
            [
                f"answer is for {answer.responds_to.decision_id}, "
                f"not {request.decision_id}"
            ]
        )
    if answer.responds_to.request_version != request.request_version:
        raise AnswerStale(
            [
                "answer is for question version "
                f"{answer.responds_to.request_version}; the current version is "
                f"{request.request_version}"
            ]
        )
    if answer.action not in request.actions:
        raise AnswerRejected([f"action {answer.action!r} is not allowed here"])

    result: Dict[str, Any] = {
        "decision_id": request.decision_id,
        "answer_id": answer.answer_id,
        "action": answer.action,
    }
    if answer.comment:
        result["comment"] = answer.comment
    if answer.action == "submit":
        result["answer"] = _check_fields(request.fields, answer.answer, "")
    return result


def check_value(field: Any, value: Any, path: str, partial: bool = False) -> Any:
    """Check one field's answer (for_each, count, type, profile); return it cleaned.

    ``partial`` allows a for_each value to leave items out: a default may
    pre-fill only the rows it knows.
    """
    if not field.for_each:
        return _check_counted(field, value, path)
    if not isinstance(value, dict):
        raise AnswerRejected([f"{path}: expected an answer per item, keyed by id"])
    open_ids = [item.id for item in field.for_each if not item.unavailable]
    problems = (
        []
        if partial
        else [f"{path}.{i}: no answer" for i in open_ids if i not in value]
    )
    problems += [f"{path}.{i}: not an item here" for i in value if i not in open_ids]
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
        raise AnswerRejected([f"{path}: expected a list"])
    low, high = field.count.min, field.count.max
    if len(value) < low or (high is not None and len(value) > high):
        limit = f"{low} or more" if high is None else f"between {low} and {high}"
        raise AnswerRejected([f"{path}: {len(value)} given; give {limit}"])
    cleaned, problems = [], []
    for index, entry in enumerate(value):
        try:
            cleaned.append(_check_single(field, entry, f"{path}[{index}]"))
        except AnswerRejected as error:
            problems += error.problems
    if field.type == "choice" and not problems:
        ids = [entry["id"] for entry in cleaned]
        if len(set(ids)) != len(ids):
            problems.append(f"{path}: the same option was picked twice")
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
            raise AnswerRejected([f"{path}: {error}"]) from None
    return cleaned


def _check_choice(field: ChoiceField, value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("id"), str):
        raise AnswerRejected([f'{path}: expected {{"id": ...}}'])
    extra = set(value) - {"id", "fields"}
    if extra:
        raise AnswerRejected([f"{path}: unexpected keys {sorted(extra)}"])
    option = next((o for o in field.options if o.id == value["id"]), None)
    if option is None:
        raise AnswerRejected([f"{path}: unknown option {value['id']!r}"])
    if option.unavailable:
        raise AnswerRejected(
            [f"{path}: option {option.id!r} is unavailable ({option.unavailable})"]
        )
    if not option.fields:
        if value.get("fields"):
            raise AnswerRejected([f"{path}: option {option.id!r} has no fields"])
        return {"id": option.id}
    sub = value.get("fields")
    if not isinstance(sub, dict):
        raise AnswerRejected([f"{path}.fields: option {option.id!r} needs its fields"])
    return {
        "id": option.id,
        "fields": _check_fields(option.fields, sub, f"{path}.fields"),
    }


def _check_number(field: NumberField, value: Any, path: str) -> Union[int, float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnswerRejected([f"{path}: expected a number"])
    if field.integer and not float(value).is_integer():
        raise AnswerRejected([f"{path}: {value} must be a whole number"])
    if field.min is not None and value < field.min:
        raise AnswerRejected([f"{path}: {value} is below min ({field.min})"])
    if field.max is not None and value > field.max:
        raise AnswerRejected([f"{path}: {value} is above max ({field.max})"])
    return int(value) if field.integer else value


def _check_text(field: TextField, value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnswerRejected([f"{path}: expected non-empty text"])
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
    prefix = f"{path}." if path else ""
    if not isinstance(answer, dict):
        raise AnswerRejected([f"{path or 'answer'}: expected an answer per field"])
    by_key = {field.key: field for field in fields}
    problems: List[str] = []
    for key, field in by_key.items():
        optional = field.count is not None and field.count.min == 0
        if key not in answer and not optional:
            problems.append(f"{prefix}{key}: no answer")
    problems += [
        f"{prefix}{key}: not a field in this question"
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
    if problems:
        raise AnswerRejected(problems)
    return cleaned


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
        if field.for_each and field.view != "plate_map":
            raise ValueError(f"{where}: for_each needs view 'plate_map' here")
        if field.count and field.type != "choice":
            raise ValueError(f"{where}: count is only supported on a choice here")
        if field.type == "choice" and any(o.fields for o in field.options):
            raise ValueError(f"{where}: options with fields aren't supported here")


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
