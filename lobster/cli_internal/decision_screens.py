"""Decision screens: ask_user decision requests in, screen data out, and
screen replies back into the contract's answers.

Translates between the decision contract (``component_schemas``) and simple
screen data (the Go TUI's BioComp components; the classic UI reuses the same
data), in both directions:

- a field becomes one or more renders, each ``(component, data)`` for a Go
  component (choice, choice_list, threshold_slider, text_input, plate_map,
  ontology_browser, sequence_view, dashboard);
- each component's own reply shape becomes the contract's answer shape.

Go components keep their own reply shapes; only this module knows them. The
launcher supplies ``render``, which shows one component and waits for its
reply, so this module can be tested without a Go process. The classic UI
supplies a text-prompt ``render`` and reuses the same translation.

A re-asked request (one with ``problems``) shows the problems and starts from
the fields' defaults; they count as the user's previous answers only when
``prefilled_from_answer`` is true. An older Go TUI that
can't draw a component shows its text box with ``fallback_prompt`` instead;
its ``{"answer": text}`` reply becomes the ``free_text`` answer action.
"""

from __future__ import annotations

import math
import re
import uuid
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lobster.services.interaction.component_schemas import (
    MAX_ATTEMPTS,
    DecisionRequest,
    _is_one_required_text,
    plain_number,
    well_index,
)
from lobster.services.interaction.view_sources import SUGGESTED

# render(component, data) -> (action, reply). action is "submit", "cancel" or
# "defer"; reply is the component's own data.
Render = Callable[[str, Dict[str, Any]], Tuple[str, Dict[str, Any]]]

_PLATE_COLUMNS = 12  # a 96-well plate is 8 rows x 12 columns
_PLATE_WELLS = 96  # the plate is always drawn as 96 wells
_PLATE_CHOICES = 9  # one number key (1-9) per choice
_LARGEST_WHOLE = 2**53  # the largest whole number a float holds exactly

# The key each component's reply carries. A reply without it but with
# "answer" came from an older TUI's text box (see _said).
_REPLY_KEY = {
    "choice": "selected",
    "choice_list": "answers",
    "plate_map": "assignments",
    "threshold_slider": "value",
    "dashboard": "answers",
    "ontology_browser": "selected",
    "sequence_view": "selected",
}

# Components that reply like "choice": {"selected": [{"id": ...}]}.
_PICKS = ("choice", "ontology_browser", "sequence_view")

# 2,000,000 or -1,234.5: commas only between groups of three digits, as
# numbers are shown; "1,5" stays "not a number".
_THOUSANDS = re.compile(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?")
_WRITE_IN_YOUR_OWN_WORDS = "Write your answer in your own words."


class Stopped(Exception):
    """The user cancelled or deferred part-way through a question."""

    def __init__(self, action: str):
        self.action = action
        super().__init__(action)


class _FreeText(Exception):
    """An older TUI showed the text box; the user answered in their own words."""

    def __init__(self, text: str):
        self.text = text
        super().__init__(text)


def unsupported(request: Dict[str, Any], go_screens: bool = True) -> List[str]:
    """What this terminal can't draw in a request. Empty when all is drawable.

    With ``go_screens`` (the Go TUI), it also lists the shapes the Go screens
    refuse to start with, so they are never sent to fail there. The classic
    UI asks in text and passes False.
    """
    parsed = DecisionRequest.model_validate(request)
    problems: List[str] = []

    def visit(fields: List[Any], path: str) -> None:
        for field in fields:
            where = f"{path}{field.key}"
            if field.count is not None and field.type != "choice":
                problems.append(f"{where}: a list of {field.type} values (count)")
            if go_screens:
                problems.extend(f"{where}: {limit}" for limit in _go_limits(field))
            if field.type == "record":
                visit(field.fields, f"{where}.")
            if field.type == "choice":
                for option in field.options:
                    if option.fields:
                        visit(option.fields, f"{where}.{option.id}.")

    visit(parsed.fields, "")
    return problems


def _go_limits(field: Any) -> List[str]:
    """The limits of the Go screens that ``field`` breaks (their Init refuses).

    lobster-tui/internal/biocomp: the plate takes 1 to 9 choices
    (sampleplate.go) and, drawn as 96 wells, at most 96 samples (unavailable
    ones take a well too, greyed); a plate needs one sample that can be
    assigned and a choice_list one available item (choicelist.go); a slider
    needs min below max (threshold.go).
    """
    limits = []
    if (
        field.type == "choice"
        and field.for_each
        and (_draws_plate(field) or _is_list(field))
    ):
        kind = "plate" if _draws_plate(field) else "list"
        open_items = [item for item in field.for_each if not item.unavailable]
        if not open_items:
            limits.append(f"a {kind} where every item is unavailable")
        if _draws_plate(field):
            choices = [option for option in field.options if not option.unavailable]
            if len(choices) > _PLATE_CHOICES:
                limits.append(
                    f"a plate with {len(choices)} options (at most {_PLATE_CHOICES})"
                )
            if len(field.for_each) > _PLATE_WELLS:
                limits.append(
                    f"a plate with {len(field.for_each)} samples (at most {_PLATE_WELLS})"
                )
    if (
        field.type == "number"
        and field.view == "slider"
        and field.min is not None
        and field.max is not None
        and field.min >= field.max
    ):
        limits.append(f"a slider whose min and max are both {plain_number(field.min)}")
    return limits


def cannot_show(problems: Sequence[str]) -> str:
    """The cancel comment for a request this terminal can't draw."""
    return (
        "Not answered: this terminal can't show "
        + "; ".join(problems)
        + ". Don't ask again with the same shape; ask in chat or with a shape "
        "it can show."
    )


def collect(request: Dict[str, Any], render: Render) -> Tuple[str, Dict[str, Any]]:
    """Ask a whole request; return (action, answer in the contract's shape).

    For "free_text" the answer is ``{"text": <the user's own words>}``.
    """
    parsed = DecisionRequest.model_validate(request)
    try:
        if parsed.view == "dashboard":
            return "submit", _collect_dashboard(parsed, render)
        notes = _problems_by_field(parsed)
        first = parsed.fields[0].key
        notes[first] = _reask_banner(parsed) + notes.pop("", []) + notes.get(first, [])
        return "submit", {
            field.key: _ask(field, render, notes.get(field.key, []))
            for field in parsed.fields
        }
    except Stopped as stop:
        return stop.action, {}
    except _FreeText as said:
        return "free_text", {"text": said.text}


def build_answer(
    request: Dict[str, Any],
    action: str,
    answer: Optional[Dict[str, Any]] = None,
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """The answer envelope for a request, as sent back through the resume."""
    envelope: Dict[str, Any] = {
        "schema": request["schema"],
        "responds_to": {
            "decision_id": request["decision_id"],
            "request_version": request["request_version"],
        },
        "answer_id": f"ans-{uuid.uuid4()}",
        "action": action,
    }
    if action == "submit":
        envelope["answer"] = answer or {}
    elif action == "free_text":
        envelope["text"] = (answer or {}).get("text", "")
    if comment:
        envelope["comment"] = comment
    return envelope


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------


def _collect_dashboard(request: DecisionRequest, render: Render) -> Dict[str, Any]:
    """One dashboard render; each section's reply becomes that field's answer."""
    replies = _show(render, "dashboard", dashboard_data(request)).get("answers") or {}
    answer = {}
    for field in request.fields:
        if field.key not in replies:
            continue  # the contract check reports it as missing
        raw = replies[field.key]
        try:
            answer[field.key] = _from_reply(field, *_render_one(field), raw)
        except ValueError:
            # Not a number: pass the text on, so the check names the field.
            answer[field.key] = str((raw or {}).get("answer", ""))
    return answer


def _ask(field: Any, render: Render, notes: Sequence[str] = ()) -> Any:
    """Ask a field (all of its for_each items); return its answer."""
    if field.for_each and (_draws_plate(field) or _is_list(field)):
        return _shown(field, render, *_render_one(field, notes=notes))
    if field.for_each:
        return {
            item.id: _ask_one(field, render, item, notes)
            for item in field.for_each
            if not item.unavailable
        }
    return _ask_one(field, render, None, notes)


def _ask_one(field: Any, render: Render, item: Any, notes: Sequence[str]) -> Any:
    """Ask a field once (for one for_each item, or for the field itself)."""
    if field.type == "record":
        earlier = _item_default(field, item)
        return {
            sub.key: _ask(_follow_up(sub, field, item, None, earlier), render)
            for sub in field.fields
        }
    component, data = _render_one(field, item, notes)
    value = _shown(field, render, component, data)
    if field.type != "choice":
        return value
    # An option with fields asks its follow-up fields once it is picked.
    picks = value if isinstance(value, list) else [value]
    for pick in picks:
        if not isinstance(pick, dict):
            continue  # nothing picked: the contract check reports it
        # An id the question doesn't offer has no follow-up fields; the
        # contract check names it and the question is asked again.
        option = next((o for o in field.options if o.id == pick.get("id")), None)
        if option is not None and option.fields:
            earlier = _picked_fields(_item_default(field, item), option.id)
            pick["fields"] = {
                sub.key: _ask(_follow_up(sub, field, item, option, earlier), render)
                for sub in option.fields
            }
    return value


def _follow_up(sub: Any, parent: Any, item: Any, option: Any, earlier: Any) -> Any:
    """A sub-field as its own screen: its question names the item and the
    parent (or the option picked), and an earlier answer is its default."""
    lead = option.label if option is not None else _plain_question(parent)
    lead = f"{lead}: " if item is None else f"{_item_name(item)}, {lead}: "
    update = {"question": lead + _plain_question(sub)}
    if isinstance(earlier, dict) and earlier.get(sub.key) is not None:
        update["default"] = earlier[sub.key]
    return sub.model_copy(update=update)


def _picked_fields(default: Any, option_id: str) -> Dict[str, Any]:
    """The fields of ``option_id`` in a default (one pick or a list)."""
    picks = default if isinstance(default, list) else [default]
    for pick in picks:
        if isinstance(pick, dict) and pick.get("id") == option_id:
            return pick.get("fields") or {}
    return {}


def _shown(field: Any, render: Render, component: str, data: Dict[str, Any]) -> Any:
    """Render until the reply translates to a valid value, or the user stops."""
    first = data
    while True:
        reply = _show(render, component, data)
        try:
            return _from_reply(field, component, data, reply)
        except ValueError as error:
            # Only typed-in numbers can fail here; ask again with the reason
            # above the whole question (its details and bounds).
            data = {**first, "question": f"{error}\n{first.get('question', '')}"}


def _show(render: Render, component: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Render once and return the submitted reply.

    Raises Stopped for cancel/defer and _FreeText for an older TUI's text-box
    reply. An empty text-box reply shows the question again.
    """
    while True:
        action, reply = render(component, data)
        if action != "submit":
            raise Stopped(action)
        said = _said(component, reply)
        if said is None:
            return reply or {}
        if said:
            raise _FreeText(said)


def _said(component: str, reply: Any) -> Optional[str]:
    """The text of an older TUI's text-box reply to ``component``, else None.

    Such a TUI draws a text box (with the fallback prompt) for a component it
    doesn't know and replies ``{"answer": text}``.
    """
    key = _REPLY_KEY.get(component)
    if key is None or not isinstance(reply, dict):
        return None
    if key in reply or "answer" not in reply:
        return None
    return str(reply["answer"] or "").strip()


def _draws_plate(field: Any) -> bool:
    """A plate_map field is drawn as a plate when each option asks for at
    most one required text field (a reason, typed on the plate) and none is
    unavailable: the plate's legend has no place for an option shown greyed
    with its reason. Any other such field is drawn as the per-item list (FX
    W1). The view is a drawing hint; the answer is the same."""
    return (
        field.type == "choice"
        and bool(field.for_each)
        and field.view == "plate_map"
        and not any(option.unavailable for option in field.options)
        and all(
            not option.fields or _is_one_required_text(option.fields)
            for option in field.options
        )
    )


def _is_list(field: Any) -> bool:
    """A per-item choice drawn as one choice_list: one option per item, and
    each option has no fields or one required text field (a reason)."""
    return (
        field.type == "choice"
        and bool(field.for_each)
        and not _draws_plate(field)
        and field.view not in ("ontology_browser", "sequence")
        and field.count is None
        and all(
            not option.fields or _is_one_required_text(option.fields)
            for option in field.options
        )
    )


# ---------------------------------------------------------------------------
# Field -> Go component data
# ---------------------------------------------------------------------------


def _render_one(
    field: Any, item: Any = None, notes: Sequence[str] = ()
) -> Tuple[str, Dict[str, Any]]:
    """The Go component and data for one field (for one item, if given).

    ``notes`` are problems from a re-ask, shown as "⚠ <message>" lines first
    in the details.
    """
    question = _item_question(field, item)
    details = _join(
        *[f"⚠ {note}" for note in notes],
        item.details if item else None,
        field.details,
    )
    default = _item_default(field, item)

    if _draws_plate(field):
        return "plate_map", _plate_data(field, _join(question, details))
    if item is None and _is_list(field):
        return "choice_list", _list_data(field, question, details)
    if field.type == "choice" and field.view == "ontology_browser":
        return "ontology_browser", _ontology_data(field, question, details, default)
    if field.type == "choice" and field.view == "sequence":
        return "sequence_view", _sequence_data(field, question, details, default)
    if field.type == "choice":
        count = field.count
        data: Dict[str, Any] = {
            "question": question,
            "options": [_item(option) for option in field.options],
            "min": count.min if count else 1,
            "max": (
                (count.max if count.max is not None else len(field.options))
                if count
                else 1
            ),
        }
        picked = default if isinstance(default, list) else [default] if default else []
        if picked:
            data["default"] = [pick["id"] for pick in picked]
        return "choice", _with(data, "details", details)
    if field.type == "number" and field.view == "slider":
        data = {
            "label": _join(question, details, sep=" · "),
            "min": field.min,
            "max": field.max,
            "default": default if default is not None else field.min,
        }
        if field.integer:
            data["step"] = _whole_step(field.min, field.max, data["default"])
        else:
            data["step"] = _round_step(field.min, field.max, data["default"])
        return "threshold_slider", _with(data, "unit", field.unit)
    # text, and numbers without a slider (typed in, parsed here).
    hint = _number_hint(field) if field.type == "number" else None
    data = {"question": _join(question, details, hint, sep="\n")}
    if default is not None:
        data["placeholder"] = f"Enter = keep {default}"
    return "text_input", data


def _whole_step(low: float, high: float, default: Any) -> int:
    """A round step (1, 2, 5, 10, 20, 50, ...) for a whole-number slider,
    about a hundredth of the range (shift moves ten steps), on which the
    proposed value sits exactly. The slider snaps to min + k * step, so a
    step of 48 from 200 showed a proposed 500 as 488 and never reached
    1,000; 200 to 5,000 with 500 proposed now gets 50. A step that also
    divides min keeps round numbers reachable."""
    values = [low, high, default]
    if any(not float(v).is_integer() for v in values):
        return 1
    low, default = int(low), int(default)
    # The nearest round step to a hundredth of the range, a little above
    # it allowed: 48 rounds to 50, not down to 20 (240 key presses).
    target = max(1.0, 1.5 * (high - low) / 100)
    steps, scale = [], 1
    while scale <= target:
        steps += [step for step in (scale, 2 * scale, 5 * scale) if step <= target]
        scale *= 10
    steps = sorted(set(steps), reverse=True)
    for fits in (lambda s: low % s == 0, lambda s: True):
        for step in steps:
            if (default - low) % step == 0 and fits(step):
                return step
    return 1


def _round_step(low: float, high: float, default: Any = None) -> float:
    """A round step for a decimal slider, from the decimals in its bounds
    and its proposed value: 0.50 and 0.99 give 0.01, so 0.75 can be reached
    (a hundredth of the range, 0.0049, could not), and a proposed 0.025
    gives 0.001, so it isn't moved. It is made ten times finer until the
    slider has at least 20 positions, so 0 to 1 gets 0.01, not 1."""
    places = max(
        0,
        *(
            -Decimal(str(bound)).normalize().as_tuple().exponent
            for bound in (low, high, default)
            if bound is not None
        ),
    )
    step = Decimal(1).scaleb(-places)
    span = Decimal(str(high)) - Decimal(str(low))
    while span / step < 20:
        step = step.scaleb(-1)
    return float(step)


def _plate_data(field: Any, question: str) -> Dict[str, Any]:
    """Each item becomes a well: at its own position when items have them,
    else filled row by row (A1, A2, ...) and the plate says so. Unavailable
    items sit greyed in their wells with the reason ("excluded"). An option
    that asks a reason carries the key and question of its text field.
    """
    defaults = field.default or {}
    placed = all(item.position for item in field.for_each)
    wells = []
    for index, item in enumerate(field.for_each):
        if placed:
            row, col = well_index(item.position)
        else:
            row, col = divmod(index, _PLATE_COLUMNS)
        well: Dict[str, Any] = {
            "row": row,
            "col": col,
            "sample_id": item.id,
            "label": _label(item),
        }
        details = _details_map(item.details)
        if item.unavailable:
            well["excluded"] = item.unavailable
        else:
            picked = defaults.get(item.id) or {}
            well["condition"] = picked.get("id", "")
            reason = next(iter((picked.get("fields") or {}).values()), None)
            _with(well, "reason", reason)
            _with(well, "warning", item.warning)  # the reason hint
        wells.append(_with(well, "details", details))
    options = []
    for option in field.options:
        data: Dict[str, Any] = {"id": option.id, "label": option.label}
        if option.fields:  # one required text field: the reason
            sub = option.fields[0]
            data["reason"] = _with({"key": sub.key}, "question", sub.question)
        options.append(data)
    return {
        "question": question,
        "title": field.question or field.key,
        "format": _PLATE_WELLS,  # unsupported() refuses more samples
        "positions": "given" if placed else "list_order",
        "options": options,
        "wells": wells,
    }


def _ontology_data(
    field: Any, question: str, details: str, default: Any
) -> Dict[str, Any]:
    """The ontology's terms as a tree (each names its parent), the
    suggested ones listed first on the screen, and the default term."""
    options = []
    for option in field.options:
        data = _item(option)
        options.append(_with(data, "parent", option.parent))
    data = {
        "question": question,
        "ontology": field.options_from.ontology if field.options_from else "",
        "options": options,
        "suggested": [
            option.id
            for option in field.options
            if (option.details or "").startswith(SUGGESTED)
        ],
    }
    if isinstance(default, dict) and default.get("id"):
        data["default"] = default["id"]
    return _with(data, "details", details)


def _sequence_data(
    field: Any, question: str, details: str, default: Any
) -> Dict[str, Any]:
    """The sequence's letters, and each region as an option with its
    start and end (1-based, inclusive)."""
    source = field.sequence_from
    options = []
    for option in field.options:
        data = _item(option)
        if option.start is not None:
            data.update(start=option.start, end=option.end)
        options.append(data)
    data = {
        "question": question,
        "name": source.id or source.file,
        "kind": source.kind,
        "sequence": source.sequence,
        "options": options,
    }
    if isinstance(default, dict) and default.get("id"):
        data["default"] = default["id"]
    return _with(data, "details", details)


# "batch: B1 · note: re-plated" (as items_from writes details).
_DETAIL_PART = re.compile(r"([^:·]{1,40}):\s*(.+)")


def _details_map(text: Optional[str]) -> Dict[str, str]:
    """An item's details as the plate shows them: "key: value" parts joined
    by " · " keep their keys; other text is a note."""
    if not text:
        return {}
    parts = [part.strip() for part in text.split(" · ")]
    found = [_DETAIL_PART.fullmatch(part) for part in parts]
    if not all(found):
        return {"note": text}
    return {match.group(1).strip(): match.group(2).strip() for match in found}


def _list_data(field: Any, question: str, details: str) -> Dict[str, Any]:
    """One row per item (unavailable ones shown greyed), one option per row."""
    options = []
    for option in field.options:
        data = _item(option)
        if option.fields:  # one required text field: the reason
            sub = option.fields[0]
            data["reason"] = _with({"key": sub.key}, "question", sub.question)
        options.append(data)
    data = {
        "question": question,
        "items": [_item(item) for item in field.for_each],
        "options": options,
    }
    if field.default:
        data["default"] = {item_id: dict(row) for item_id, row in field.default.items()}
    return _with(data, "details", details)


def dashboard_data(request: DecisionRequest) -> Dict[str, Any]:
    """One section per field, each with the component that draws it.

    On a re-ask (the request has problems), each problem goes to the section
    its path starts with (others above the sections). Only when the defaults
    are the user's submitted answer (``prefilled_from_answer``) does every
    section with one get it as ``answer``, in that component's reply shape,
    so Go starts it answered; the supervisor's proposals only pre-fill.
    """
    notes = _problems_by_field(request)
    sections = []
    for field in request.fields:
        component, data = _render_one(field)
        section = {
            "key": field.key,
            "title": field.question or field.key,
            "depends_on": field.depends_on,
            "component": component,
            "data": data,
        }
        if notes.get(field.key):
            section["problems"] = notes[field.key]
        if request.prefilled_from_answer and field.default is not None:
            section["answer"] = _as_reply(field, component, field.default)
        sections.append(section)
    out: Dict[str, Any] = {"question": request.question}
    top = _reask_banner(request) + notes.get("", [])
    if top:
        out["problems"] = top
    out["sections"] = sections
    return out


def _reask_banner(request: DecisionRequest) -> List[str]:
    """On a re-ask, one line saying so, so it doesn't look like the first
    screen; on the last attempt, that a refusal closes the question."""
    version = request.request_version
    if version <= 1:
        return []
    line = f"Asked again (attempt {version} of {MAX_ATTEMPTS}): fix what is marked"
    line += "; your earlier answers are kept." if request.prefilled_from_answer else "."
    if version >= MAX_ATTEMPTS:
        line += (
            " This is the last attempt: if this answer is refused too, the"
            " question is closed and nothing is decided."
        )
    return [line]


def _problems_by_field(request: DecisionRequest) -> Dict[str, List[str]]:
    """Problem lines by the field their path starts with ("" for the rest).

    A problem deeper in a field (an item, a reason) names where, e.g.
    "S02 reason: expected non-empty text".
    """
    by_key = {field.key: field for field in request.fields}
    out: Dict[str, List[str]] = {}
    for problem in request.problems:
        head, rest = re.match(r"([^.\[]*)(.*)", problem.path).groups()
        if head not in by_key:
            out.setdefault("", []).append(problem.message)
            continue
        # An item id may itself hold a "." (ENSG00000141510.17): match it
        # against the field's items before splitting the rest.
        item = next(
            (
                i.id
                for i in sorted(by_key[head].for_each or [], key=lambda i: -len(i.id))
                if rest == f".{i.id}" or rest.startswith((f".{i.id}.", f".{i.id}["))
            ),
            None,
        )
        if item is not None:
            rest = rest[len(item) + 1 :]
        parts = [p for p in re.split(r"[.\[\]]+", rest) if p and p != "fields"]
        where = " ".join(([item] if item is not None else []) + parts)
        out.setdefault(head, []).append(
            f"{where}: {problem.message}" if where else problem.message
        )
    return out


# ---------------------------------------------------------------------------
# Plain-text fallback: the whole question in words
# ---------------------------------------------------------------------------


def fallback_prompt(request: Dict[str, Any]) -> str:
    """The whole question in plain words, for a TUI that can only show text.

    Question, problems, then each field (each dashboard section, in order):
    its question, details, items and options ("id — label: details",
    "(unavailable: reason)"), then how to answer in one's own words.
    """
    parsed = DecisionRequest.model_validate(request)
    numbered = len(parsed.fields) > 1
    lines = [parsed.question] + _reask_banner(parsed)
    for key, notes in _problems_by_field(parsed).items():
        where = f"{key}: " if key and numbered else ""
        lines += [f"⚠ {where}{note}" for note in notes]
    for number, field in enumerate(parsed.fields, 1):
        lines.append("")
        head = field.question or field.key
        if numbered:
            lines.append(f"{number}. {head}")
        elif head != parsed.question:
            lines.append(head)
        lines += _describe(field, "   " if numbered else "")
    return "\n".join(lines + ["", _WRITE_IN_YOUR_OWN_WORDS])


def _describe(field: Any, indent: str) -> List[str]:
    """Plain lines saying what a field asks and what may be answered."""
    lines = [f"{indent}{field.details}"] if field.details else []
    if field.for_each:
        lines.append(f"{indent}For each of:")
        lines += [f"{indent}- {_plain(item)}" for item in field.for_each]
    if field.type == "choice":
        count = field.count
        if count is None:
            lines.append(f"{indent}Pick one{' per item' if field.for_each else ''}:")
        else:
            most = f" and at most {count.max}" if count.max is not None else ""
            lines.append(f"{indent}Pick at least {count.min}{most}:")
        for option in field.options:
            asks = ""
            if option.fields:
                names = ", ".join(sub.question or sub.key for sub in option.fields)
                asks = f" (then give: {names})"
            lines.append(f"{indent}- {_plain(option)}{asks}")
    elif field.type == "number":
        lines.append(f"{indent}{_number_hint(field)}.")
    elif field.type == "text":
        lines.append(f"{indent}Answer in text.")
    elif field.type == "record":
        for sub in field.fields:
            lines.append(f"{indent}- {sub.question or sub.key}:")
            lines += _describe(sub, indent + "  ")
    return lines


def _plain(item: Any) -> str:
    """An item as "id — label: details (unavailable: reason)", minus what is unset."""
    text = item.id if item.label == item.id else f"{item.id} — {item.label}"
    if item.details:
        text += f": {item.details}"
    if item.unavailable:
        text += f" (unavailable: {item.unavailable})"
    return text


# ---------------------------------------------------------------------------
# Go component reply <-> contract answer
# ---------------------------------------------------------------------------


def _from_reply(field: Any, component: str, data: Dict[str, Any], reply: Any) -> Any:
    """Translate one component's own reply into the field's answer shape."""
    reply = reply or {}
    if component == "plate_map":
        # A reason comes back keyed by sample id; it becomes the fields of the
        # option that asked for it, the shape the per-item list sends.
        keys = {o["id"]: o["reason"]["key"] for o in data["options"] if o.get("reason")}
        reasons = reply.get("reasons") or {}
        answers = {}
        for item, choice in (reply.get("assignments") or {}).items():
            answers[item] = {"id": choice}
            if choice in keys and reasons.get(item):
                answers[item]["fields"] = {keys[choice]: reasons[item]}
        return answers
    if component == "choice_list":
        answers = {}
        for item, row in (reply.get("answers") or {}).items():
            row = row or {}
            answers[item] = {"id": row.get("id")}
            if row.get("fields"):
                answers[item]["fields"] = dict(row["fields"])
        return answers
    if component in _PICKS:
        picks = [{"id": pick["id"]} for pick in reply.get("selected") or []]
        if field.count is None:
            return picks[0] if picks else None
        return picks
    if component == "threshold_slider":
        value = reply.get("value")
        return int(value) if field.integer and value is not None else value
    text = str(reply.get("answer", "")).strip()
    if not text and "placeholder" in data:  # Enter on an empty box keeps the default
        return field.default
    if field.type != "number":
        return text
    plain = text.replace(",", "") if _THOUSANDS.fullmatch(text) else text
    try:
        number = float(plain)
    except ValueError:
        raise ValueError(f"'{text}' is not a number.") from None
    if not math.isfinite(number):  # "nan", "inf", 1e309
        raise ValueError(f"'{text}' is not a number.")
    if field.integer:
        if not number.is_integer():
            raise ValueError(f"{text} is not a whole number.")
        if abs(number) > _LARGEST_WHOLE:
            # Beyond it a float is no longer exact, and past 64 bits the
            # graph's checkpoint can't store the answer at all.
            raise ValueError(f"{text} is too large.")
        return int(number)
    return number


def _as_reply(field: Any, component: str, value: Any) -> Dict[str, Any]:
    """A field's answer in ``component``'s own reply shape (the reverse of
    _from_reply), so a re-asked dashboard section can start from it."""
    if component == "plate_map":
        reply = {"assignments": {item: row["id"] for item, row in value.items()}}
        reasons = {
            item: next(iter(row["fields"].values()))
            for item, row in value.items()
            if row.get("fields")
        }
        return _with(reply, "reasons", reasons)
    if component == "choice_list":
        return {"answers": {item: dict(row) for item, row in value.items()}}
    if component in _PICKS:
        picks = value if isinstance(value, list) else [value]
        return {"selected": [{"id": pick["id"]} for pick in picks]}
    if component == "threshold_slider":
        return {"value": value}
    return {"answer": str(value)}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _item(option: Any) -> Dict[str, Any]:
    data = {"id": option.id, "label": option.label}
    warning = getattr(option, "warning", None)
    _with(data, "details", _join(option.details, warning and f"⚠ {warning}", sep=" · "))
    return _with(data, "unavailable", option.unavailable)


def _label(item: Any) -> str:
    """The item's label, or "" when it is just the id."""
    return item.label if item.label != item.id else ""


def _item_question(field: Any, item: Any) -> str:
    question = _plain_question(field)
    if item is None:
        return question
    return f"{_item_name(item)}: {question}"


def _item_name(item: Any) -> str:
    """ "label (id)"; the label alone when it already holds the id."""
    return item.label if item.id in item.label else f"{item.label} ({item.id})"


def _plain_question(field: Any) -> str:
    return field.question or field.key


def _item_default(field: Any, item: Any) -> Any:
    if item is None or field.default is None:
        return field.default
    return field.default.get(item.id)


def _number_hint(field: Any) -> str:
    kind = "a whole number" if field.integer else "a number"
    bounds = []
    if field.min is not None:
        bounds.append(f"min {plain_number(field.min)}")
    if field.max is not None:
        bounds.append(f"max {plain_number(field.max)}")
    unit = f" ({field.unit})" if field.unit else ""
    return f"Enter {kind}{unit}" + (f", {', '.join(bounds)}" if bounds else "")


def _join(*parts: Optional[str], sep: str = "\n") -> str:
    return sep.join(part for part in parts if part)


def _with(data: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    if value:
        data[key] = value
    return data
