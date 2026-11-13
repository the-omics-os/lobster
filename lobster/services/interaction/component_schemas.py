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

import csv
import hashlib
import math
import os
import re
import stat
import sys
import uuid
from collections import Counter
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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

SCHEMA_VERSION = "lobster.ask/1"

# After this many refused answers in a row the supervisor is told instead
# ("rejected"), so an interface or a user that keeps sending a bad answer
# can't loop. Surfaces use it to say which attempt the user is on.
MAX_ATTEMPTS = 5

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


def plain_number(value: Union[int, float]) -> str:
    """A number as a person reads it: 2,000,000 or 0.31, never 2e+06."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.10f}".rstrip("0").rstrip(".")


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
    warning: Optional[str] = Field(
        default=None,
        description="A caution about this item (a QC flag, an unreadable label); "
        "shown beside it, and offered on a plate as the reason (never filled in)",
    )
    position: Optional[str] = Field(
        default=None,
        description='Where a sample physically sits, a plate well such as "D1" '
        '(or {"row": "D", "column": 1}); a plate draws it there',
    )

    @field_validator("position", mode="before")
    @classmethod
    def _position_is_a_well(cls, value: Any) -> Any:
        """ "D1", "d01" and {"row": "D", "column": 1} all become "D1"."""
        if value is None:
            return None
        if isinstance(value, dict):
            if set(value) != {"row", "column"}:
                raise ValueError('a position is "D1" or {"row": "D", "column": 1}')
            value = f"{value['row']}{value['column']}"
        if not isinstance(value, str):
            raise ValueError(f"position {value!r} is not a well like 'D1'")
        found = _WELL.fullmatch(value.strip().upper())
        if not found:
            raise ValueError(f"position {value!r} is not a well like 'D1'")
        return found.group(1) + found.group(2)

    @model_validator(mode="before")
    @classmethod
    def _accept_plain_string(cls, value: Any) -> Any:
        """Allow "S01" as shorthand for {"id": "S01"}."""
        if isinstance(value, str):
            return {"id": value}
        return value

    @field_validator("id")
    @classmethod
    def _id_is_exact(cls, value: str) -> str:
        """No spaces at either end: "S01" and "S01 " would look identical."""
        if value != value.strip() or not value.strip():
            raise ValueError(f"id {value!r} has spaces at an end or is blank")
        return value

    @model_validator(mode="after")
    def _label_defaults_to_id(self) -> "Item":
        if not self.label:
            self.label = self.id
        return self


class Option(Item):
    """A choice option. ``fields`` appear only when this option is picked.

    ``parent`` (view "ontology_browser") places the option under another
    option in a tree; ``start``/``end`` (view "sequence") make it a region of
    the sequence, 1-based and inclusive. Python sets ``parent`` from the
    ontology; the supervisor writes ``start``/``end``.
    """

    fields: Optional[List["QuestionField"]] = Field(
        default=None,
        min_length=1,
        description="Extra fields shown only when this option is picked",
    )
    parent: Optional[str] = Field(
        default=None, description="The option this one sits under (ontology tree)"
    )
    start: Optional[int] = Field(
        default=None, ge=1, description="First position of a sequence region (1-based)"
    )
    end: Optional[int] = Field(
        default=None, ge=1, description="Last position of a sequence region (inclusive)"
    )
    removes: Optional[bool] = Field(
        default=None,
        description="True: picking it takes the item out (it must ask a reason). "
        "False: it doesn't, whatever its name sounds like",
    )

    @model_validator(mode="after")
    def _check_option_fields(self) -> "Option":
        if self.position is not None:
            raise ValueError(
                "an option has no position; positions are for for_each items"
            )
        if self.fields:
            _check_siblings(self.fields)
        if (self.start is None) != (self.end is None):
            raise ValueError(f"option {self.id!r} needs both start and end, or neither")
        if self.start is not None and self.start > self.end:
            raise ValueError(
                f"option {self.id!r} starts at {self.start}, after its end {self.end}"
            )
        if self.parent is not None and self.parent == self.id:
            raise ValueError(f"option {self.id!r} can't be its own parent")
        return self


# A plate well: one or two row letters, then the column number ("D1", "AA12").
_WELL = re.compile(r"([A-Z]{1,2})0*([1-9][0-9]{0,2})")
PLATE_ROWS, PLATE_COLUMNS = 8, 12  # a plate is drawn as 96 wells, A1 to H12


def well_index(position: str) -> Tuple[int, int]:
    """(row, column) of a well, from 0: "D1" is (3, 0), "AA1" is (26, 0)."""
    found = _WELL.fullmatch(position)
    if not found:
        raise ValueError(f"position {position!r} is not a well like 'D1'")
    letters, number = found.groups()
    row = 0
    for letter in letters:
        row = row * 26 + (ord(letter) - ord("A") + 1)
    return row - 1, int(number) - 1


class ItemsFrom(BaseModel):
    """A choice field's ``items_from``: its items (and their proposed options)
    are rows of a CSV or TSV file in the workspace, so the supervisor names
    columns instead of copying ids by hand. Python reads the file when the
    question is built; the screens get the same question as with for_each."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1, description="Workspace-relative .csv or .tsv")
    id: str = Field(min_length=1, description="Column with each item's stable id")
    label: Optional[str] = Field(default=None, description="Column shown as the label")
    default: Optional[str] = Field(
        default=None,
        description="Column with each item's proposed option; a value that is "
        "not an option (blank, '?', 'unknown') proposes nothing",
    )
    details: List[str] = Field(
        default_factory=list, description="Columns shown beside each item"
    )
    position: Optional[str] = Field(
        default=None, description='Column with each sample\'s well, e.g. "D1"'
    )
    warning: Optional[str] = Field(
        default=None, description="Column with each item's warning, if any"
    )
    unavailable: Optional["ItemsFromUnavailable"] = Field(
        default=None,
        description="Rows that can't be answered, shown greyed with a reason: "
        '{"column": "sample_type", "values": {"blank": "blank"}}',
    )
    where: Dict[str, str] = Field(
        default_factory=dict, description="Keep only rows with these column values"
    )
    match: Literal["exact", "ignore_case_and_spaces"] = Field(
        default="exact",
        description="How 'where' values and 'default' values are compared",
    )


class ItemsFromUnavailable(BaseModel):
    """Rows whose ``column`` holds one of ``values`` (compared with the
    table's ``match``) become unavailable items with that value's reason:
    on a plate they sit greyed in their wells."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1)
    values: Dict[str, Annotated[str, Field(min_length=1)]] = Field(min_length=1)


class OptionsFrom(BaseModel):
    """A choice field's ``options_from``: its options are terms of a public
    ontology, so ids come from the ontology, never typed by the LLM. The
    supervisor names the ontology and suggests terms by name (or id); Python
    loads the ontology when the question is built and fills ``options`` with
    the suggestions and their neighbourhood (see view_sources)."""

    model_config = ConfigDict(extra="forbid")

    ontology: str = Field(min_length=1, description='e.g. "uberon", "cell_ontology"')
    suggest: List[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1,
        max_length=10,
        description="Suggested terms, by exact name or id; shown first",
    )


class SequenceFrom(BaseModel):
    """A choice field's ``sequence_from``: the sequence its regions are on,
    read from a FASTA file in the workspace by Python. ``sequence`` and
    ``kind`` are filled in by ask_user, never by the supervisor."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1, description="Workspace-relative FASTA file")
    id: Optional[str] = Field(
        default=None, description="Record id; needed when the file has several"
    )
    sequence: Optional[str] = Field(
        default=None, description="The letters, read from the file by ask_user"
    )
    kind: Optional[Literal["dna", "rna", "protein"]] = Field(
        default=None, description="Read from the letters by ask_user"
    )


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
            "key": "max_missing",
            "type": "number",
            "view": "slider",
            "min": 0,
            "max": 1,
            "default": 0.3,
        },
    },
    "plate_map": {
        "applies_to": "choice",
        "needs": ["for_each"],
        "use_when": "putting samples into groups, one group each, shown as wells "
        "on a plate (up to 96 samples and 9 options; an option may ask one "
        "reason); ask over all the samples, known groups pre-filled; for more, "
        "or items that aren't samples, leave the view out: a per-item list",
        "example": {
            "key": "condition",
            "type": "choice",
            "view": "plate_map",
            "for_each": ["S01", "S02", "S03"],
            "options": ["control", "treated"],
        },
        "check": lambda owner: _check_plate_map(owner),
    },
    "ontology_browser": {
        "applies_to": "choice",
        "needs": ["options_from"],
        "use_when": "picking ontology terms (tissue, cell type, disease), one "
        "per item with for_each or one in all; options_from names the ontology "
        '("uberon", "cell_ontology", "mondo") and suggests terms by exact name; '
        "Python fills the options with real ids, so leave options out; the user "
        "may pick a nearby term instead",
        "example": {
            "key": "tissue",
            "type": "choice",
            "view": "ontology_browser",
            "for_each": ["S01", "S02"],
            "options_from": {"ontology": "uberon", "suggest": ["liver", "lung"]},
        },
        "check": lambda owner: _check_ontology_browser(owner),
    },
    "sequence": {
        "applies_to": "choice",
        "needs": ["sequence_from"],
        "use_when": "picking one region of a sequence (a guide, primer, ORF); "
        "sequence_from names the FASTA file (and the record id if it has "
        "several), each option is a region with start and end (1-based, "
        "inclusive); Python reads the letters, never copy them",
        "example": {
            "key": "guide",
            "type": "choice",
            "view": "sequence",
            "sequence_from": {"file": "data/construct.fasta", "id": "construct_1"},
            "options": [
                {"id": "g1", "label": "guide 1", "start": 120, "end": 139},
                {"id": "g2", "label": "guide 2", "start": 410, "end": 429},
            ],
        },
        "check": lambda owner: _check_sequence(owner),
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

    @field_validator("key")
    @classmethod
    def _key_is_a_plain_name(cls, value: str) -> str:
        """Problem paths join keys with "." and "[i]", so a key can't hold them."""
        if value != value.strip() or any(c in value for c in ".[]"):
            raise ValueError(
                f"key {value!r} can't contain '.', '[' or ']' or spaces at an end"
            )
        return value

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
    items_from: Optional[ItemsFrom] = Field(
        default=None,
        description="Where for_each was read from (set by ask_user when it "
        "reads the table); the table is read again if it changes",
    )
    options_from: Optional[OptionsFrom] = Field(
        default=None,
        description="The ontology the options were filled from (view "
        "ontology_browser)",
    )
    sequence_from: Optional[SequenceFrom] = Field(
        default=None,
        description="The sequence the regions are on (view sequence)",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_id_default(cls, data: Any) -> Any:
        """Read a default naming a known option by its bare id as {"id": ...}.

        Deliberate, pinned leniency (U-17), for the supervisor's input only:
        "SRC-A", {"S02": "retain"} and ["S05", "S07"] become the full form,
        like an Item written as a plain string. Answers get no shorthand.
        Before-mode, so the default's own check and the request-time checks
        see the full form, and so does the wire.
        """
        if not isinstance(data, dict) or data.get("default") is None:
            return data
        known = set()
        for option in data.get("options") or []:
            if isinstance(option, dict):
                option = option.get("id")
            option = getattr(option, "id", option)
            if isinstance(option, str):
                known.add(option)

        def full(value: Any) -> Any:
            if data.get("count") is not None and isinstance(value, list):
                return [full_one(entry) for entry in value]
            return full_one(value)

        def full_one(value: Any) -> Any:
            return {"id": value} if isinstance(value, str) and value in known else value

        default = data["default"]
        if data.get("for_each") and isinstance(default, dict):
            default = {item: full(value) for item, value in default.items()}
        else:
            default = full(default)
        return {**data, "default": default}

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
        # These keys only mean something on their own view; elsewhere they
        # would be silently ignored, so they are refused.
        if self.view != "ontology_browser" and any(o.parent for o in self.options):
            raise ValueError("an option's parent is only for view 'ontology_browser'")
        if self.view != "sequence" and any(o.start for o in self.options):
            raise ValueError("an option's start and end are only for view 'sequence'")
        if self.options_from is not None and self.view != "ontology_browser":
            raise ValueError("options_from is only for view 'ontology_browser'")
        if self.sequence_from is not None and self.view != "sequence":
            raise ValueError("sequence_from is only for view 'sequence'")
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
        # Compared as numbers: "10" and "9" written as text would compare as
        # text, and "0" against 1 would crash. A bound that isn't a number at
        # all is left to the field's own check, which names it.
        if isinstance(data, dict):
            low, high = _as_number(data.get("min")), _as_number(data.get("max"))
            if low is not None and high is not None and low > high:
                raise ValueError(
                    f"min ({data['min']}) is larger than max ({data['max']})"
                )
        return data


def _as_number(value: Any) -> Optional[float]:
    """``value`` as a number (from a number or numeric text), else None."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
        duplicates = sorted(p for p, n in Counter(self.data_files).items() if n > 1)
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
        description="Hex digest; None for a quick fingerprint, or in a fresh "
        "reading of a file that is gone",
    )
    quick: Optional[str] = Field(
        default=None,
        description="Set instead of sha256 for a file fingerprinted by size and time "
        "(above QUICK_FINGERPRINT_BYTES, or FOLDER_QUICK_BYTES in the data/ watch): "
        '"<size> bytes, modified <mtime_ns>" (fingerprinted by size and time)',
    )

    @model_serializer(mode="wrap")
    def _leave_out_unset_quick(self, handler: Any) -> Dict[str, Any]:
        """A hashed file's fingerprint stays {"path", "sha256"}, as before."""
        data = handler(self)
        if data.get("quick") is None:
            data.pop("quick", None)
        return data


class FileSnapshot(BaseModel):
    """What the decision's data files looked like when the question was asked."""

    model_config = ConfigDict(extra="forbid")

    files: List[FileFingerprint] = Field(min_length=1)
    folder: bool = Field(
        default=False,
        description="True when the files are the data/ folder watch (no "
        "data_files named): a deleted file there is reported once, then "
        "forgotten, so an unrelated deletion never blocks the answer",
    )

    @model_serializer(mode="wrap")
    def _leave_out_named(self, handler: Any) -> Dict[str, Any]:
        """A named-files snapshot stays {"files"}, as before."""
        data = handler(self)
        if not data.get("folder"):
            data.pop("folder", None)
        return data


_CHUNK = 1024 * 1024

# Fingerprints: a full sha256 is exact but reads every byte (about 280 MB/s
# here, so 1 GB is ~4 s, when asking and again on submit); size and
# modification time is instant but misses a change that keeps both.
#
# The files the question names (data_files) are what the decision is about:
# read in full up to this size, by size and time above it, so a huge .h5ad
# or BAM never freezes a question for minutes.
QUICK_FINGERPRINT_BYTES = 1024 * 1024 * 1024

# The data/ folder watch is only a safety net for a question that named no
# files, over up to DATA_FOLDER_LIMIT files: only files up to this size (a
# sample sheet, a small table) are read in full, so the whole watch stays
# around a second or two (500 x 1 MB at most).
FOLDER_QUICK_BYTES = 1024 * 1024

# The most files the default watch (the workspace's data/ folder) records.
DATA_FOLDER_LIMIT = 500


def data_folder_files(
    workspace: Union[str, Path], limit: int = DATA_FOLDER_LIMIT
) -> List[str]:
    """Workspace-relative paths of the regular files in the workspace's data/
    folder and its subfolders, in a stable order: hidden files and folders
    and links are skipped, and the list stops at ``limit``. Empty when the
    folder is missing."""
    root = Path(workspace).resolve()
    folder = root / "data"
    if folder.is_symlink() or not folder.is_dir():
        return []
    found: List[str] = []
    for current, folders, names in os.walk(folder):  # links are not followed
        folders[:] = sorted(name for name in folders if not name.startswith("."))
        for name in sorted(names):
            path = Path(current) / name
            if name.startswith(".") or path.is_symlink() or not path.is_file():
                continue
            found.append(path.relative_to(root).as_posix())
            if len(found) >= limit:
                return found
    return found


def watch_data_folder(workspace: Union[str, Path]) -> Optional[FileSnapshot]:
    """The fingerprints of the data/ folder's files, for a question that
    named no data_files; None when there is nothing to watch. A file that
    can't be read is left out rather than stopping the question."""
    paths = data_folder_files(workspace)
    if not paths:
        return None
    readable = [
        entry
        for entry in fingerprint(
            paths, workspace, missing_ok=True, quick_above=FOLDER_QUICK_BYTES
        ).files
        if _seen(entry) is not None
    ]
    return FileSnapshot(files=readable, folder=True) if readable else None


def _seen(entry: FileFingerprint) -> Optional[Tuple[Optional[str], Optional[str]]]:
    """What a fingerprint recorded, or None for a file that is gone."""
    if entry.sha256 is None and entry.quick is None:
        return None
    return entry.sha256, entry.quick


def fingerprint(
    paths: Iterable[str],
    workspace: Union[str, Path],
    missing_ok: bool = False,
    quick_above: Optional[int] = None,
) -> FileSnapshot:
    """sha256 of each workspace file, read in chunks; for a file above
    ``quick_above`` (default QUICK_FINGERPRINT_BYTES), its size and
    modification time instead.

    Paths must be relative and stay inside the workspace (no absolute paths,
    no "..", no links out). A file that breaks this, or is missing or
    unreadable, raises ValueError, or with ``missing_ok`` gets
    ``sha256 = None`` (used when re-reading files at answer time, where any
    such failure means the data changed).
    """
    root = Path(workspace).resolve()
    limit = QUICK_FINGERPRINT_BYTES if quick_above is None else quick_above
    return _fingerprints(root, [(raw, limit) for raw in paths], missing_ok)


def refingerprint(snapshot: FileSnapshot, workspace: Union[str, Path]) -> FileSnapshot:
    """The snapshot's files read again the same way each was recorded (in
    full, or by size and time), so a file is always compared like with like,
    whichever limit chose its kind when the question was asked. A file that
    is gone or unreadable gets neither (it counts as changed)."""
    root = Path(workspace).resolve()
    every_size = sys.maxsize  # never above it: read in full
    return _fingerprints(
        root,
        [
            (entry.path, -1 if entry.quick is not None else every_size)
            for entry in snapshot.files
        ],
        missing_ok=True,
    )


def _fingerprints(
    root: Path, paths: List[Tuple[str, int]], missing_ok: bool
) -> FileSnapshot:
    files = []
    for raw, limit in paths:
        relative = Path(raw)
        try:
            digest, quick = _fingerprint_inside(root, relative, raw, limit)
        except (OSError, RuntimeError, ValueError):
            # RuntimeError: a link loop on Python < 3.13.
            if not missing_ok:
                raise
            digest = quick = None
        files.append(
            FileFingerprint(path=relative.as_posix(), sha256=digest, quick=quick)
        )
    return FileSnapshot(files=files)


def _fingerprint_inside(
    root: Path, relative: Path, raw: str, limit: int
) -> Tuple[Optional[str], Optional[str]]:
    """(sha256, None), or (None, size and time) for a file above ``limit``;
    ValueError if the file is outside the workspace or unreadable."""
    resolved = _file_inside(root, relative, raw)
    try:
        status = resolved.stat()
        if status.st_size > limit:
            return None, f"{status.st_size} bytes, modified {status.st_mtime_ns}"
        return _sha256(resolved), None
    except OSError as error:
        reason = error.strerror or type(error).__name__
        raise ValueError(f"can't read {raw!r} in the workspace ({reason})") from error


def _file_inside(root: Path, relative: Path, raw: str) -> Path:
    """``root / relative`` resolved; ValueError unless it is a regular file
    inside the workspace."""
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"{raw!r} must be a path inside the workspace (relative, without '..')"
        )
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{raw!r} leads outside the workspace")
    try:
        # A pipe or device would block the read forever: regular files only.
        if not stat.S_ISREG(resolved.stat().st_mode):
            raise ValueError(f"can't read {raw!r} in the workspace (not a file)")
    except OSError as error:
        reason = error.strerror or type(error).__name__
        raise ValueError(f"can't read {raw!r} in the workspace ({reason})") from error
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Items from a table: for_each and default read from a workspace file
# ---------------------------------------------------------------------------

# The most rows items_from turns into items: a plate draws 96 wells; a list
# of more than this many is no longer a question one person answers.
ITEMS_FROM_MAX_ROWS = 500
_PLATE_WELLS = PLATE_ROWS * PLATE_COLUMNS
_TABLE_ROWS_READ = 100_000  # stop reading a table larger than this


def expand_items_from(
    context: Any, workspace: Optional[Union[str, Path]]
) -> Tuple[Any, List[Problem]]:
    """Replace each top-level field's ``items_from`` with the ``for_each`` and
    ``default`` read from its file, and add the file to ``data_files``.

    Returns the new context, or the problems that stop it (a missing file or
    column, a blank or duplicate id, too many or no rows), so nothing built
    from a broken table is ever shown. A context without items_from is
    returned unchanged.
    """
    fields = context.get("fields") if isinstance(context, dict) else None
    if not isinstance(fields, list) or not any(
        isinstance(field, dict) and "items_from" in field for field in fields
    ):
        return context, []
    problems: List[Problem] = []
    expanded: List[Any] = []
    files = list(context.get("data_files") or [])
    for index, field in enumerate(fields):
        if not (isinstance(field, dict) and "items_from" in field):
            expanded.append(field)
            continue
        where = f"fields.{index}.items_from"
        built, found = _expand_field(field, where, workspace)
        problems += found
        expanded.append(built)
        file = field["items_from"].get("file") if not found else None
        if file and file not in files:
            files.append(file)
    if problems:
        return context, problems
    return {**context, "fields": expanded, "data_files": files}, []


def _expand_field(
    field: Dict[str, Any], where: str, workspace: Optional[Union[str, Path]]
) -> Tuple[Dict[str, Any], List[Problem]]:
    try:
        spec = ItemsFrom.model_validate(field["items_from"])
    except ValidationError as error:
        return field, [
            _problem(
                ".".join([where, *(str(part) for part in item["loc"])]),
                item["msg"].removeprefix("Value error, "),
            )
            for item in error.errors()
        ]
    if field.get("type") != "choice":
        return field, [_problem(where, "items_from is only for a choice field")]
    if field.get("for_each"):  # an empty for_each counts as left out
        return field, [_problem(where, "give for_each or items_from, not both")]
    if field.get("default") is not None:
        return field, [
            _problem(
                where,
                "items_from proposes each item's option from its 'default' "
                "column; leave the field's default out",
            )
        ]
    if workspace is None:
        return field, [
            _problem(where, "no workspace is open, so the file can't be read")
        ]
    items, defaults, problems = read_items(
        spec, field.get("options"), field.get("view"), workspace, where
    )
    if problems:
        return field, problems
    built = dict(field)
    # Kept with the question, so a changed table can be read again (reread_items).
    built["items_from"] = spec.model_dump(exclude_none=True)
    built["for_each"] = items
    if defaults:
        built["default"] = defaults
    return built, []


def read_items(
    spec: ItemsFrom,
    options: Any,
    view: Optional[str],
    workspace: Union[str, Path],
    where: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Problem]]:
    """The items and proposed options a table gives, or the problems with it."""
    try:
        header, rows = _read_table(Path(workspace).resolve(), spec.file)
    except ValueError as error:
        return [], {}, [_problem(f"{where}.file", str(error))]

    gone = spec.unavailable.column if spec.unavailable else None
    wanted = [spec.id, spec.label, spec.default, spec.position, spec.warning, gone]
    missing = [
        column
        for column in dict.fromkeys([*wanted, *spec.details, *spec.where])
        if column and column not in header
    ]
    if missing:
        return (
            [],
            {},
            [
                _problem(
                    where,
                    f"{spec.file} has no column {', '.join(repr(c) for c in missing)}; "
                    f"its columns are {', '.join(repr(c) for c in header)}",
                )
            ],
        )

    same = _matcher(spec.match)
    rows = [
        row
        for row in rows
        if all(same(row.get(column, ""), value) for column, value in spec.where.items())
    ]
    plate = view == "plate_map"
    limit = _PLATE_WELLS if plate else ITEMS_FROM_MAX_ROWS
    if not rows:
        return [], {}, [_problem(where, f"no rows of {spec.file} match {spec.where}")]
    if len(rows) > limit:
        advice = (
            "a plate holds at most 96 samples; leave the view out to ask them "
            "as a list"
            if plate
            else f"one question takes at most {limit}"
        )
        return (
            [],
            {},
            [_problem(where, f"{len(rows)} rows of {spec.file} match; {advice}")],
        )

    ids = [(row.get(spec.id) or "").strip() for row in rows]
    blank = [str(number) for number, item_id in enumerate(ids, 1) if not item_id]
    if blank:
        return (
            [],
            {},
            [_problem(where, f"rows {_some(blank)} have no {spec.id!r} (the id)")],
        )
    repeated = sorted(item_id for item_id, n in Counter(ids).items() if n > 1)
    if repeated:
        return (
            [],
            {},
            [
                _problem(
                    where, f"{spec.id!r} repeats ids {_some(repeated)} in {spec.file}"
                )
            ],
        )

    names = _option_names(options)
    items: List[Dict[str, Any]] = []
    defaults: Dict[str, Any] = {}
    for item_id, row in zip(ids, rows):
        item: Dict[str, Any] = {"id": item_id}

        def value(column: Optional[str], row: Dict[str, str] = row) -> str:
            return (row.get(column) or "").strip() if column else ""

        if value(spec.label):
            item["label"] = value(spec.label)
        shown = [f"{c}: {value(c)}" for c in spec.details if value(c)]
        if shown:
            item["details"] = " · ".join(shown)
        if value(spec.warning):
            item["warning"] = value(spec.warning)
        if value(spec.position):
            item["position"] = value(spec.position)
        if spec.unavailable:
            reason = next(
                (
                    why
                    for match, why in spec.unavailable.values.items()
                    if same(row.get(gone, ""), match)
                ),
                None,
            )
            if reason:
                item["unavailable"] = reason
        items.append(item)
        if spec.default and "unavailable" not in item:
            picked = [
                o
                for o, alias in names
                if any(same(row[spec.default], n) for n in alias)
            ]
            if len(picked) == 1:  # one option: proposed; none or several: unresolved
                defaults[item_id] = {"id": picked[0]}
    return items, defaults, []


def _read_table(root: Path, raw: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """The header and rows of a workspace .csv or .tsv file."""
    relative = Path(raw)
    suffix = relative.suffix.lower()
    if suffix not in (".csv", ".tsv", ".tab"):
        raise ValueError(f"{raw!r} is not a .csv or .tsv file")
    path = _file_inside(root, relative, raw)
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter="," if suffix == ".csv" else "\t")
            header = [column for column in reader.fieldnames or []]
            rows = []
            for row in reader:
                rows.append(row)
                if len(rows) > _TABLE_ROWS_READ:
                    raise ValueError(f"{raw!r} has more than {_TABLE_ROWS_READ} rows")
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"can't read {raw!r} ({type(error).__name__})") from error
    if not header:
        raise ValueError(f"{raw!r} is empty")
    return header, rows


def _matcher(match: str) -> Callable[[str, str], bool]:
    if match == "ignore_case_and_spaces":
        return (
            lambda a, b: "".join(str(a).split()).casefold()
            == "".join(str(b).split()).casefold()
        )
    return lambda a, b: str(a).strip() == str(b).strip()


def _option_names(options: Any) -> List[Tuple[str, List[str]]]:
    """Each option's id with the names a table may use for it: id and label."""
    named = []
    for option in options or []:
        if isinstance(option, str):
            named.append((option, [option]))
        elif isinstance(option, dict) and isinstance(option.get("id"), str):
            names = [option["id"]]
            if isinstance(option.get("label"), str) and option["label"]:
                names.append(option["label"])
            if not option.get("unavailable"):
                named.append((option["id"], names))
    return named


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
    current = {entry.path: _seen(entry) for entry in now.files}
    return [
        entry.path
        for entry in asked.files
        if current.get(entry.path) is None or current[entry.path] != _seen(entry)
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
    problems: List[Problem] = _removals_without_reason(context.fields, "")
    for spec in context.checks:
        entry = CHECKS[spec["check"]]
        if entry["on"] == "request":
            problems += entry["run"](spec, by_key, None)
    return problems


# An option whose id or label sounds like it takes an item out of the
# analysis. Only a safety net for options the supervisor didn't mark with
# "removes"; the mark always wins.
_TAKES_OUT = re.compile(
    r"\b(remov\w*|exclu\w*|drop\w*|discard\w*|omit\w*|leave\s*out|left\s*out"
    r"|not\s*included)\b"
)


def _removals_without_reason(fields: List[Any], path: str) -> List[Problem]:
    """Options that take something out must ask for a reason, whatever checks
    the supervisor attached: a removal is never recorded without one.

    An option marked "removes": true is one; "removes": false is not, whatever
    its name. An unmarked option is one when its id or label sounds like it
    (remove, exclude, drop, discard, omit, leave out, not included), and the
    problem says how to mark it if it isn't. Unavailable options are skipped
    (they can't be picked); nested fields are checked too."""
    problems: List[Problem] = []
    for field in fields:
        where = f"{path}{field.key}"
        if field.type == "record":
            problems += _removals_without_reason(field.fields, f"{where}.")
        if field.type != "choice":
            continue
        for option in field.options:
            if option.fields:
                problems += _removals_without_reason(
                    option.fields, f"{where}.{option.id}."
                )
            if option.unavailable or _asks_a_reason(option) or option.removes is False:
                continue
            if option.removes:
                problems.append(
                    _problem(
                        where,
                        f"option '{option.id}' takes something out (removes: true): "
                        f"give it a reason field, {REASON_FIELD}",
                    )
                )
            elif sounds_like_removal(option):
                problems.append(
                    _problem(
                        where,
                        f"option '{option.id}' sounds like it takes something out: "
                        f"give it a reason field, {REASON_FIELD}, or mark it "
                        '"removes": false if it doesn\'t',
                    )
                )
    return problems


def sounds_like_removal(option: Any) -> bool:
    """The option's id or label sounds like it takes an item out."""
    words = f"{option.id} {option.label}".replace("_", " ").replace("-", " ")
    return bool(_TAKES_OUT.search(words.lower()))


# How a reason is written, shown wherever one is missing.
REASON_FIELD = '"fields": [{"key": "reason", "type": "text"}]'


def _is_reason(sub: Any) -> bool:
    """A reason box: one text answer (no count, which would make the answer a
    list; no for_each). The one definition the removal rule, the
    reason_required check and the screens all use."""
    return sub.type == "text" and sub.count is None and not sub.for_each


def _asks_a_reason(option: Any) -> bool:
    """The option asks for a reason (other fields may sit beside it)."""
    return any(_is_reason(sub) for sub in option.fields or [])


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
    is_open = set(open_ids)
    problems = (
        []
        if partial
        else [_problem(f"{path}.{i}", "no answer") for i in open_ids if i not in value]
    )
    problems += [
        _problem(f"{path}.{i}", "not an item here") for i in value if i not in is_open
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
    try:
        finite = math.isfinite(value)
    except OverflowError:  # an int too large for a float
        raise AnswerRejected([_problem(path, "the number is too large")]) from None
    if not finite:  # NaN passes every bound, and is not valid JSON
        raise AnswerRejected([_problem(path, "expected a finite number")])
    if field.integer and not float(value).is_integer():
        raise AnswerRejected(
            [_problem(path, f"{plain_number(value)} must be a whole number")]
        )
    if field.min is not None and value < field.min:
        raise AnswerRejected(
            [
                _problem(
                    path,
                    f"{plain_number(value)} is below min ({plain_number(field.min)})",
                )
            ]
        )
    if field.max is not None and value > field.max:
        raise AnswerRejected(
            [
                _problem(
                    path,
                    f"{plain_number(value)} is above max ({plain_number(field.max)})",
                )
            ]
        )
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
    as it was. Otherwise it becomes True only when every default is now the
    user's: refilled from ``submitted``, or already the user's before. A
    field left out of ``submitted`` (an empty answer, a partial submit)
    keeps the supervisor's default, which is never shown as the user's,
    so then it is False. ``snapshot``, a fresh reading of the data files,
    replaces the old fingerprints, except for a file that is gone. A named
    file that is gone keeps its old fingerprint, so the answer stays stale
    until the file is back (the decision is about it). A gone file in the
    data/ folder watch is dropped: it was reported once, and an unrelated
    deletion must not block the answer forever.
    """
    fields = request.fields
    prefilled = request.prefilled_from_answer
    if isinstance(submitted, dict):
        refilled = [_refill(field, submitted) for field in request.fields]
        fields = [field for field, _ in refilled]
        owned = [
            from_user or request.prefilled_from_answer
            for field, from_user in refilled
            if field.default is not None
        ]
        prefilled = bool(owned) and all(owned)
    update: Dict[str, Any] = {
        "request_version": request.request_version + 1,
        "problems": list(problems),
        "fields": fields,
        "prefilled_from_answer": prefilled,
    }
    if snapshot is not None and request.snapshot is not None:
        fresh = {
            entry.path: entry for entry in snapshot.files if _seen(entry) is not None
        }
        old = request.snapshot
        if old.folder:
            kept = [fresh[entry.path] for entry in old.files if entry.path in fresh]
            update["snapshot"] = (
                old.model_copy(update={"files": kept}) if kept else None
            )
        else:
            update["snapshot"] = old.model_copy(
                update={"files": [fresh.get(e.path, e) for e in old.files]}
            )
    return request.model_copy(update=update)


def reread_items(
    request: DecisionRequest,
    before: Optional[FileSnapshot],
    workspace: Optional[Union[str, Path]],
) -> DecisionRequest:
    """After a stale answer, read every table a field came from again.

    The re-ask then shows the table as it is now: positions and items come
    from the new file, answers are kept for ids still there, new rows start
    unresolved, and vanished ids are named in a problem. If a table can no
    longer be read, the field keeps its old items, a problem says why, and
    that file keeps its old fingerprint (``before``), so an answer stays
    stale until the table is fixed.
    """
    if workspace is None or not any(
        getattr(field, "items_from", None) for field in request.fields
    ):
        return request
    fields: List[Any] = []
    problems = list(request.problems)
    unreadable = set()
    for field in request.fields:
        spec = getattr(field, "items_from", None)
        if spec is None:
            fields.append(field)
            continue
        options = [o.model_dump(exclude_none=True) for o in field.options]
        items, _, found = read_items(spec, options, field.view, workspace, field.key)
        rebuilt = None
        if not found:
            open_ids = {item["id"] for item in items if "unavailable" not in item}
            kept = {k: v for k, v in (field.default or {}).items() if k in open_ids}
            try:
                rebuilt = type(field).model_validate(
                    {
                        **field.model_dump(exclude_none=True),
                        "for_each": items,
                        "default": kept or None,
                    }
                )
            except ValidationError as error:
                found = problems_of(error, field.key)
        if rebuilt is None:
            problems += [
                _problem(field.key, f"couldn't read {spec.file} again: {p.message}")
                for p in found
            ]
            unreadable.add(spec.file)
            fields.append(field)
            continue
        old = [item.id for item in field.for_each]
        new = [item["id"] for item in items]
        vanished = [item_id for item_id in old if item_id not in new]
        added = [item_id for item_id in new if item_id not in old]
        if vanished:
            problems.append(
                _problem(field.key, f"no longer in {spec.file}: {_some(vanished)}")
            )
        if added:
            problems.append(
                _problem(
                    field.key,
                    f"new in {spec.file}, not answered yet: {_some(added)}",
                )
            )
        fields.append(rebuilt)
    update: Dict[str, Any] = {"fields": fields, "problems": problems}
    if unreadable and before is not None and request.snapshot is not None:
        old = {entry.path: entry for entry in before.files}
        update["snapshot"] = request.snapshot.model_copy(
            update={
                "files": [
                    old.get(entry.path, entry) if entry.path in unreadable else entry
                    for entry in request.snapshot.files
                ]
            }
        )
    return request.model_copy(update=update)


def problems_of(error: ValidationError, path: str) -> List[Problem]:
    """One Problem per pydantic error, under ``path``."""
    return [
        _problem(path, item["msg"].removeprefix("Value error, "))
        for item in error.errors()
    ]


def _refill(field: Any, submitted: Dict[str, Any]) -> Tuple[Any, bool]:
    """``field`` with the submitted value as its default, where it is valid.

    Also says whether the new default is all the user's: False when the
    field kept its previous default, or a for_each field kept rows the
    user didn't send.
    """
    if field.key not in submitted:
        return field, False
    value = submitted[field.key]
    if field.for_each:
        if not isinstance(value, dict):
            return field, False
        rows = dict(field.default) if isinstance(field.default, dict) else {}
        sent = set()
        for item_id, row in value.items():
            try:
                rows.update(check_value(field, {item_id: row}, field.key, partial=True))
                sent.add(item_id)
            except AnswerRejected:
                rows.pop(item_id, None)
        refilled = field.model_copy(update={"default": rows or None})
        return refilled, set(rows) <= sent
    try:
        default = check_value(field, value, field.key, partial=True)
    except AnswerRejected:
        return field, False
    return field.model_copy(update={"default": default}), True


# ---------------------------------------------------------------------------
# Helpers for request validation
# ---------------------------------------------------------------------------


def _check_unique_ids(items: List[Item], where: str) -> None:
    duplicates = sorted(
        i for i, n in Counter(item.id for item in items).items() if n > 1
    )
    if duplicates:
        raise ValueError(f"{where} have duplicate ids {duplicates}")


def _check_siblings(fields: List[Any]) -> None:
    """Sibling keys are unique; depends_on names existing siblings, without cycles."""
    duplicates = sorted(k for k, n in Counter(f.key for f in fields).items() if n > 1)
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


def _check_plate_map(field: Any) -> None:
    """A well holds one option, so the plate can't answer a count. Positions
    are given for every item or none, each inside the plate, one per well."""
    if field.count is not None:
        raise ValueError(
            "view 'plate_map' takes one option per item; leave count out "
            "(or leave the view out to pick several per item)"
        )
    items = field.for_each or []
    unplaced = [item.id for item in items if item.position is None]
    if len(unplaced) == len(items):
        return
    if unplaced:
        raise ValueError(
            f"give every item a position, or none; {_some(unplaced)} have none"
        )
    taken: Dict[str, str] = {}
    for item in items:
        row, column = well_index(item.position)
        if row >= PLATE_ROWS or column >= PLATE_COLUMNS:
            raise ValueError(
                f"item {item.id!r} is at {item.position}, outside the plate "
                f"(A1 to H{PLATE_COLUMNS})"
            )
        if item.position in taken:
            raise ValueError(
                f"items {taken[item.position]!r} and {item.id!r} are both in well "
                f"{item.position}"
            )
        taken[item.position] = item.id


def _check_ontology_browser(field: Any) -> None:
    """One term per pick, no follow-up fields, and the parents form a tree
    over the options (each names another option, without a loop)."""
    if field.count is not None:
        raise ValueError(
            "view 'ontology_browser' takes one term per pick; leave count out"
        )
    if any(option.fields for option in field.options):
        raise ValueError("view 'ontology_browser' options can't have fields")
    parents = {option.id: option.parent for option in field.options}
    for option_id, parent in parents.items():
        if parent is not None and parent not in parents:
            raise ValueError(
                f"option {option_id!r} sits under {parent!r}, which is not an option"
            )
        seen = {option_id}
        while parent is not None:
            if parent in seen:
                raise ValueError(f"the options' parents loop through {parent!r}")
            seen.add(parent)
            parent = parents.get(parent)


# The longest sequence one question shows: longer ones are not read by eye.
SEQUENCE_MAX_LENGTH = 100_000


def _check_sequence(field: Any) -> None:
    """One region, picked once; the sequence was read (by ask_user); every
    region that can be picked has a start and end inside it."""
    if field.count is not None or field.for_each:
        raise ValueError(
            "view 'sequence' asks for one region; leave count and for_each out"
        )
    if any(option.fields for option in field.options):
        raise ValueError("view 'sequence' options can't have fields")
    letters = field.sequence_from.sequence
    if not letters:
        raise ValueError(
            "view 'sequence' has no sequence; name the FASTA file in "
            "sequence_from and leave sequence out (it is read from the file)"
        )
    for option in field.options:
        if option.unavailable:
            continue
        if option.start is None:
            raise ValueError(f"region {option.id!r} needs a start and an end")
        if option.end > len(letters):
            raise ValueError(
                f"region {option.id!r} ends at {option.end}, past the end of the "
                f"sequence ({len(letters)})"
            )


def _some(ids: List[str], limit: int = 5) -> str:
    """At most ``limit`` ids, then how many more: "S01, S02 and 3 more"."""
    shown = ", ".join(ids[:limit])
    return shown if len(ids) <= limit else f"{shown} and {len(ids) - limit} more"


def _check_dashboard(owner: Any) -> None:
    """Each dashboard section must be a field the dashboard can draw as one panel."""
    if len(owner.fields) < 2:
        raise ValueError("view 'dashboard' needs at least 2 fields")
    for field in owner.fields:
        where = f"dashboard section {field.key!r}"
        if field.view in ("ontology_browser", "sequence"):
            raise ValueError(
                f"{where}: view {field.view!r} can't be in a dashboard yet; ask "
                "it on its own"
            )
        if field.type == "record":
            raise ValueError(f"{where}: a record can't be a section; split it")
        if field.count and field.type != "choice":
            raise ValueError(f"{where}: count is only supported on a choice here")
        if field.for_each and field.type != "choice":
            raise ValueError(f"{where}: for_each is only supported on a choice here")
        if field.for_each:
            # One option per item; an option may ask for a reason (the list
            # and the plate both collect it).
            if field.count:
                raise ValueError(f"{where}: a per-item list takes one option per item")
            for option in field.options:
                if option.fields and not _is_one_required_text(option.fields):
                    raise ValueError(
                        f"{where}: option {option.id!r} may only have one required "
                        "text field (a reason) here"
                    )
        elif field.type == "choice" and any(o.fields for o in field.options):
            # Say why and what to do: "aren't supported here" alone made a
            # model guess twice.
            raise ValueError(
                f"{where}: in a dashboard, only a section with for_each (a list "
                "or plate) can ask a reason; ask this choice on its own, or give "
                "it for_each"
            )


def _is_one_required_text(fields: List[Any]) -> bool:
    """Exactly one reason box and nothing else: what the plate, the per-item
    list and the dashboard can draw on an option."""
    return len(fields) == 1 and _is_reason(fields[0])


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
        if not _asks_a_reason(option):
            problems.append(
                _problem(
                    field.key,
                    f"option {option.id!r} needs a required text field for the "
                    f"reason, e.g. {REASON_FIELD}",
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
