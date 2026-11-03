"""Go TUI adapter for ask_user decision requests.

Translates between the decision contract (``component_schemas``) and the Go
TUI's BioComp components, in both directions:

- a field becomes one or more renders, each ``(component, data)`` for a Go
  component (choice, threshold_slider, text_input, plate_map, dashboard);
- each component's own reply shape becomes the contract's answer shape.

Go components keep their own reply shapes; only this module knows them. The
launcher supplies ``render``, which shows one component and waits for its
reply, so this module can be tested without a Go process.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from lobster.services.interaction.component_schemas import DecisionRequest

# render(component, data) -> (action, reply). action is "submit", "cancel" or
# "defer"; reply is the component's own data.
Render = Callable[[str, Dict[str, Any]], Tuple[str, Dict[str, Any]]]

_PLATE_COLUMNS = 12  # a 96-well plate is 8 rows x 12 columns


class Stopped(Exception):
    """The user cancelled or deferred part-way through a question."""

    def __init__(self, action: str):
        self.action = action
        super().__init__(action)


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


def collect(request: Dict[str, Any], render: Render) -> Tuple[str, Dict[str, Any]]:
    """Ask a whole request; return (action, answer in the contract's shape)."""
    parsed = DecisionRequest.model_validate(request)
    if parsed.view == "dashboard":
        action, reply = render("dashboard", dashboard_data(parsed))
        if action != "submit":
            return action, {}
        replies = reply.get("answers") or {}
        answer = {}
        for field in parsed.fields:
            if field.key not in replies:
                continue  # the contract check reports it as missing
            raw = replies[field.key]
            try:
                answer[field.key] = _from_reply(field, *_render_one(field), raw)
            except ValueError:
                # Not a number: pass the text on, so the check names the field.
                answer[field.key] = str((raw or {}).get("answer", ""))
        return "submit", answer
    try:
        return "submit", {f.key: _ask(f, render) for f in parsed.fields}
    except Stopped as stop:
        return stop.action, {}


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
    if comment:
        envelope["comment"] = comment
    return envelope


# ---------------------------------------------------------------------------
# Asking one field
# ---------------------------------------------------------------------------


def _ask(field: Any, render: Render) -> Any:
    """Ask a field (all of its for_each items); return its answer."""
    if field.for_each and field.view == "plate_map":
        return _shown(field, render, *_render_one(field))
    if field.for_each:
        return {
            item.id: _ask_one(field, render, item)
            for item in field.for_each
            if not item.unavailable
        }
    return _ask_one(field, render, None)


def _ask_one(field: Any, render: Render, item: Any) -> Any:
    """Ask a field once (for one for_each item, or for the field itself)."""
    if field.type == "record":
        return {sub.key: _ask(sub, render) for sub in field.fields}
    component, data = _render_one(field, item)
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
        action, reply = render(component, data)
        if action != "submit":
            raise Stopped(action)
        try:
            return _from_reply(field, component, data, reply)
        except ValueError as error:
            # Only typed-in numbers can fail here; ask again with the reason.
            data = {**data, "question": f"{error}\n{_plain_question(field)}"}


# ---------------------------------------------------------------------------
# Field -> Go component data
# ---------------------------------------------------------------------------


def _render_one(field: Any, item: Any = None) -> Tuple[str, Dict[str, Any]]:
    """The Go component and data for one field (for one item, if given)."""
    question = _item_question(field, item)
    details = _join(item.details if item else None, field.details)
    default = _item_default(field, item)

    if field.type == "choice" and field.for_each and field.view == "plate_map":
        return "plate_map", _plate_data(field, question)
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
    """Each available item becomes a well, filled row by row (A1, A2, ...)."""
    items = [item for item in field.for_each if not item.unavailable]
    defaults = field.default or {}
    wells = []
    for index, item in enumerate(items):
        well = {
            "row": index // _PLATE_COLUMNS,
            "col": index % _PLATE_COLUMNS,
            "sample_id": item.id,
            "label": item.label if item.label != item.id else "",
            "condition": (defaults.get(item.id) or {}).get("id", ""),
        }
        if item.details:
            well["details"] = {"note": item.details}
        wells.append(well)
    return {
        "question": question,
        "title": field.question or field.key,
        "format": 96 if len(items) <= 96 else 384,
        "choices": [o.id for o in field.options if not o.unavailable],
        "wells": wells,
    }


def dashboard_data(request: DecisionRequest) -> Dict[str, Any]:
    """One section per field, each with the component that draws it."""
    sections = []
    for field in request.fields:
        component, data = _render_one(field)
        sections.append(
            {
                "key": field.key,
                "title": field.question or field.key,
                "depends_on": field.depends_on,
                "component": component,
                "data": data,
            }
        )
    return {"question": request.question, "sections": sections}


# ---------------------------------------------------------------------------
# Go component reply -> contract answer
# ---------------------------------------------------------------------------


def _from_reply(field: Any, component: str, data: Dict[str, Any], reply: Any) -> Any:
    """Translate one component's own reply into the field's answer shape."""
    reply = reply or {}
    if component == "plate_map":
        assignments = reply.get("assignments") or {}
        return {item: {"id": choice} for item, choice in assignments.items()}
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


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _item(option: Any) -> Dict[str, Any]:
    data = {"id": option.id, "label": option.label}
    _with(data, "details", option.details)
    return _with(data, "unavailable", option.unavailable)


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
