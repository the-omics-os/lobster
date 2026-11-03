"""Go TUI adapter for ask_user decision requests.

Translates between the decision contract (``component_schemas``) and the Go
TUI's BioComp components, in both directions:

- a field becomes one or more renders, each ``(component, data)`` for a Go
  component (choice, choice_list, threshold_slider, text_input, plate_map,
  dashboard);
- each component's own reply shape becomes the contract's answer shape.

Go components keep their own reply shapes; only this module knows them. The
launcher supplies ``render``, which shows one component and waits for its
reply, so this module can be tested without a Go process. The classic UI
supplies a text-prompt ``render`` and reuses the same translation.

A re-asked request (one with ``problems``) shows the problems and starts from
the user's previous answers (the fields' defaults). An older Go TUI that
can't draw a component shows its text box with ``fallback_prompt`` instead;
its ``{"answer": text}`` reply becomes the ``free_text`` answer action.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lobster.services.interaction.component_schemas import (
    DecisionRequest,
    _is_one_required_text,
)

# render(component, data) -> (action, reply). action is "submit", "cancel" or
# "defer"; reply is the component's own data.
Render = Callable[[str, Dict[str, Any]], Tuple[str, Dict[str, Any]]]

_PLATE_COLUMNS = 12  # a 96-well plate is 8 rows x 12 columns

# The key each component's reply carries. A reply without it but with
# "answer" came from an older TUI's text box (see _said).
_REPLY_KEY = {
    "choice": "selected",
    "choice_list": "answers",
    "plate_map": "assignments",
    "threshold_slider": "value",
    "dashboard": "answers",
}

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


def unsupported(request: Dict[str, Any]) -> List[str]:
    """What this terminal can't draw in a request. Empty when all is drawable."""
    parsed = DecisionRequest.model_validate(request)
    problems: List[str] = []

    def visit(fields: List[Any], path: str) -> None:
        for field in fields:
            where = f"{path}{field.key}"
            if field.count is not None and field.type != "choice":
                problems.append(f"{where}: a list of {field.type} values (count)")
            if field.type == "record":
                visit(field.fields, f"{where}.")
            if field.type == "choice":
                for option in field.options:
                    if option.fields:
                        visit(option.fields, f"{where}.{option.id}.")

    visit(parsed.fields, "")
    return problems


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
        notes[first] = notes.pop("", []) + notes.get(first, [])
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
    if field.for_each and (field.view == "plate_map" or _is_list(field)):
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
        return {sub.key: _ask(sub, render) for sub in field.fields}
    component, data = _render_one(field, item, notes)
    value = _shown(field, render, component, data)
    if field.type != "choice":
        return value
    # An option with fields asks its follow-up fields once it is picked.
    picks = value if isinstance(value, list) else [value]
    for pick in picks:
        option = next(o for o in field.options if o.id == pick["id"])
        if option.fields:
            pick["fields"] = {sub.key: _ask(sub, render) for sub in option.fields}
    return value


def _shown(field: Any, render: Render, component: str, data: Dict[str, Any]) -> Any:
    """Render until the reply translates to a valid value, or the user stops."""
    while True:
        reply = _show(render, component, data)
        try:
            return _from_reply(field, component, data, reply)
        except ValueError as error:
            # Only typed-in numbers can fail here; ask again with the reason.
            data = {**data, "question": f"{error}\n{_plain_question(field)}"}


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


def _is_list(field: Any) -> bool:
    """A per-item choice drawn as one choice_list: one option per item, and
    each option has no fields or one required text field (a reason)."""
    return (
        field.type == "choice"
        and bool(field.for_each)
        and field.view != "plate_map"
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

    if field.type == "choice" and field.for_each and field.view == "plate_map":
        return "plate_map", _plate_data(field, _join(question, details))
    if item is None and _is_list(field):
        return "choice_list", _list_data(field, question, details)
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
            data["step"] = 1
        return "threshold_slider", _with(data, "unit", field.unit)
    # text, and numbers without a slider (typed in, parsed here).
    hint = _number_hint(field) if field.type == "number" else None
    data = {"question": _join(question, details, hint, sep="\n")}
    if default is not None:
        data["placeholder"] = f"Enter = keep {default}"
    return "text_input", data


def _plate_data(field: Any, question: str) -> Dict[str, Any]:
    """Each available item becomes a well, filled row by row (A1, A2, ...).

    Unavailable items are never wells: they are listed under the plate.
    """
    items = [item for item in field.for_each if not item.unavailable]
    defaults = field.default or {}
    wells = []
    for index, item in enumerate(items):
        well = {
            "row": index // _PLATE_COLUMNS,
            "col": index % _PLATE_COLUMNS,
            "sample_id": item.id,
            "label": _label(item),
            "condition": (defaults.get(item.id) or {}).get("id", ""),
        }
        if item.details:
            well["details"] = {"note": item.details}
        wells.append(well)
    data = {
        "question": question,
        "title": field.question or field.key,
        "format": 96 if len(items) <= 96 else 384,
        "choices": [o.id for o in field.options if not o.unavailable],
        "wells": wells,
    }
    unavailable = [
        {"sample_id": item.id, "label": _label(item), "reason": item.unavailable}
        for item in field.for_each
        if item.unavailable
    ]
    return _with(data, "unavailable", unavailable)


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
    its path starts with (others above the sections), and every section with
    a previous answer gets it as ``answer``, in that component's reply shape.
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
        if request.problems and field.default is not None:
            section["answer"] = _as_reply(field, component, field.default)
        sections.append(section)
    out: Dict[str, Any] = {"question": request.question}
    if notes.get(""):
        out["problems"] = notes[""]
    out["sections"] = sections
    return out


def _problems_by_field(request: DecisionRequest) -> Dict[str, List[str]]:
    """Problem lines by the field their path starts with ("" for the rest).

    A problem deeper in a field (an item, a reason) names where, e.g.
    "S02 reason: expected non-empty text".
    """
    keys = {field.key for field in request.fields}
    out: Dict[str, List[str]] = {}
    for problem in request.problems:
        head, rest = re.match(r"([^.\[]*)(.*)", problem.path).groups()
        if head not in keys:
            out.setdefault("", []).append(problem.message)
            continue
        where = " ".join(
            part for part in re.split(r"[.\[\]]+", rest) if part and part != "fields"
        )
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
    lines = [parsed.question]
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
        assignments = reply.get("assignments") or {}
        return {item: {"id": choice} for item, choice in assignments.items()}
    if component == "choice_list":
        answers = {}
        for item, row in (reply.get("answers") or {}).items():
            row = row or {}
            answers[item] = {"id": row.get("id")}
            if row.get("fields"):
                answers[item]["fields"] = dict(row["fields"])
        return answers
    if component == "choice":
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
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a number.") from None
    if field.integer:
        if not number.is_integer():
            raise ValueError(f"{text} is not a whole number.")
        return int(number)
    return number


def _as_reply(field: Any, component: str, value: Any) -> Dict[str, Any]:
    """A field's answer in ``component``'s own reply shape (the reverse of
    _from_reply), so a re-asked dashboard section can start from it."""
    if component == "plate_map":
        return {"assignments": {item: row["id"] for item, row in value.items()}}
    if component == "choice_list":
        return {"answers": {item: dict(row) for item, row in value.items()}}
    if component == "choice":
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
    _with(data, "details", option.details)
    return _with(data, "unavailable", option.unavailable)


def _label(item: Any) -> str:
    """The item's label, or "" when it is just the id."""
    return item.label if item.label != item.id else ""


def _item_question(field: Any, item: Any) -> str:
    question = _plain_question(field)
    if item is None:
        return question
    name = item.id if item.label == item.id else f"{item.label} ({item.id})"
    return f"{name}: {question}"


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
        bounds.append(f"min {field.min:g}")
    if field.max is not None:
        bounds.append(f"max {field.max:g}")
    unit = f" ({field.unit})" if field.unit else ""
    return f"Enter {kind}{unit}" + (f", {', '.join(bounds)}" if bounds else "")


def _join(*parts: Optional[str], sep: str = "\n") -> str:
    return sep.join(part for part in parts if part)


def _with(data: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    if value:
        data[key] = value
    return data
