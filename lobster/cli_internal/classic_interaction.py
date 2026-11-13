"""Classic CLI fallback renderers for HITL interrupt components.

Used when the Go TUI is not available. Renders interrupt components as
simple terminal prompts using Python's built-in input().
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


def handle_interrupt_classic(interrupt_data: Any) -> Dict[str, Any]:
    """Render an interrupt as a terminal prompt and collect the response.

    Args:
        interrupt_data: The interrupt payload. An ask_user DecisionRequest is
                        asked as plain-text prompts (``_ask_decision``);
                        other payloads use the legacy component renderers
                        below.

    Returns:
        The value to resume the interrupt with.
    """
    if not isinstance(interrupt_data, dict):
        question = interrupt_data if isinstance(interrupt_data, str) else ""
        return _handle_text_input(question or "Please provide input", {})

    from lobster.services.interaction.component_schemas import SCHEMA_VERSION

    if interrupt_data.get("schema") == SCHEMA_VERSION:
        return _ask_decision(interrupt_data)

    component = interrupt_data.get("component", "text_input")
    fallback = interrupt_data.get("fallback_prompt", "Please provide input:")
    data = interrupt_data.get("data", {})
    if not isinstance(fallback, str) or not fallback.strip():
        fallback = "Please provide input"
    if not isinstance(data, dict):
        return _handle_text_input(fallback, {})
    if not isinstance(data.get("question"), str) or not data["question"].strip():
        data = {**data, "question": fallback}

    if component == "confirm":
        return _handle_confirm(fallback, data)
    elif component == "select":
        options = data.get("options")
        if not (
            isinstance(options, list)
            and options
            and all(isinstance(option, str) and option.strip() for option in options)
        ):
            return _handle_text_input(fallback, data)
        return _handle_select(fallback, data)
    elif component == "threshold_slider":
        return _handle_threshold(fallback, data)
    elif component == "cell_type_selector":
        return _handle_cell_type(fallback, data)
    else:
        return _handle_text_input(fallback, data)


def _handle_confirm(fallback: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Yes/no confirmation."""
    default = data.get("default", False)
    hint = "[Y/n]" if default else "[y/N]"
    question = data.get("question", fallback)
    answer = input(f"\n{question} {hint}: ").strip().lower()
    if not answer:
        confirmed = default
    else:
        confirmed = answer in ("y", "yes")
    return {"confirmed": confirmed}


def _handle_select(fallback: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Single-choice from numbered list."""
    options = data.get("options", [])
    question = data.get("question", fallback)
    print(f"\n{question}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")

    while True:
        answer = input("Enter number: ").strip()
        try:
            idx = int(answer) - 1
            if 0 <= idx < len(options):
                return {"selected": options[idx], "index": idx}
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(options)}.")


def _handle_threshold(fallback: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Numeric threshold input."""
    label = data.get("label", fallback)
    default = data.get("default", 0.5)
    min_val = data.get("min", 0.0)
    max_val = data.get("max", 1.0)
    unit = data.get("unit", "")
    unit_str = f" {unit}" if unit else ""

    print(f"\n{label}")
    print(f"  Range: {min_val}-{max_val}{unit_str}, Default: {default}{unit_str}")

    while True:
        answer = input(f"Enter value [{default}]: ").strip()
        if not answer:
            return {"value": default}
        try:
            val = float(answer)
            if min_val <= val <= max_val:
                return {"value": val}
            print(f"Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a numeric value.")


def _handle_cell_type(fallback: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Cluster cell type assignment."""
    clusters = data.get("clusters", [])
    print(f"\n{fallback}")

    assignments = {}
    for cluster in clusters:
        cid = str(cluster.get("id", "?"))
        size = cluster.get("size", "?")
        markers = ", ".join(cluster.get("markers", [])[:5])
        label = input(f"  Cluster {cid} ({size} cells, markers: {markers}): ").strip()
        if label:
            assignments[cid] = label

    return {"assignments": assignments}


def _handle_text_input(fallback: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Generic text input."""
    question = data.get("question", fallback)
    answer = input(f"\n{question}: ").strip()
    return {"answer": answer}


# ---------------------------------------------------------------------------
# ask_user DecisionRequests: structured plain-text prompts
# ---------------------------------------------------------------------------
#
# The Go adapter (go_tui_components) turns a request into component data and
# each component's reply into the contract's answer. The classic UI reuses it
# with a render that asks each component's data as text prompts and replies
# in that component's own shape, so both interfaces build the same answer and
# the contract's checks and re-asks apply unchanged.

_STOP_KEYS = {"d": "defer", "q": "cancel"}


class _Stop(Exception):
    """The user typed d (decide later) or q (cancel) at a prompt."""

    def __init__(self, action: str):
        self.action = action
        super().__init__(action)


def _ask_decision(request: Dict[str, Any]) -> Dict[str, Any]:
    """Ask an ask_user DecisionRequest; return its DecisionAnswer as a dict."""
    from lobster.cli_internal.go_tui_components import (
        build_answer,
        cannot_show,
        collect,
        unsupported,
    )

    problems = unsupported(request, go_screens=False)
    if problems:
        print(f"\n{request.get('question', '')}")
        print("This terminal can't show: " + "; ".join(problems))
        return build_answer(request, "cancel", comment=cannot_show(problems))
    print("\n(At any prompt: d = decide later, q = cancel.)")
    action, answer = collect(request, _render)
    return build_answer(request, action, answer)


def _render(component: str, data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Ask one component's data as text prompts: (action, the component's reply)."""
    try:
        return "submit", _DRAW[component](data)
    except _Stop as stop:
        return stop.action, {}


def _read(prompt: str) -> str:
    """One line from the user; d or q alone stops the question."""
    text = input(prompt).strip()
    if text.lower() in _STOP_KEYS:
        raise _Stop(_STOP_KEYS[text.lower()])
    return text


def _show(text: str, details: Optional[str] = None) -> None:
    print(f"\n{text}")
    for line in (details or "").splitlines():
        print(f"  {line}")


def _name(item: Dict[str, Any]) -> str:
    """ "label (id)"; the id when there is no label, the label when it holds the id."""
    label = item.get("label") or ""
    if not label:
        return item["id"]
    return label if item["id"] in label else f"{label} ({item['id']})"


def _list_options(options: List[Dict[str, Any]]) -> None:
    for number, option in enumerate(options, 1):
        line = f"  {number}. {_name(option)}"
        if option.get("details"):
            line += f" — {option['details']}"
        if option.get("reason"):
            line += " (asks for a reason)"
        if option.get("unavailable"):
            line += f" (unavailable: {option['unavailable']})"
        print(line)


def _pick(
    options: List[Dict[str, Any]],
    ask: str,
    default: List[int],
    low: int = 1,
    high: int = 1,
) -> List[int]:
    """Indexes of the options picked by number; asks until the input is valid.

    Empty input keeps ``default``. Unavailable options are refused.
    """
    keep = f" [Enter = {', '.join(str(i + 1) for i in default)}]" if default else ""
    while True:
        text = _read(f"{ask}{keep}: ")
        if not text:
            if default or low == 0:
                return default
            print("An answer is needed.")
            continue
        try:
            picks = [int(part) - 1 for part in text.replace(",", " ").split()]
        except ValueError:
            print("Enter the number of an option.")
            continue
        problem = _pick_problem(options, picks, low, high)
        if problem is None:
            return picks
        print(problem)


def _pick_problem(
    options: List[Dict[str, Any]], picks: List[int], low: int, high: int
) -> Optional[str]:
    for index in picks:
        if not 0 <= index < len(options):
            return f"{index + 1} is not an option (1 to {len(options)})."
        if options[index].get("unavailable"):
            option = options[index]
            return f"{option['id']} can't be picked: {option['unavailable']}."
    if len(set(picks)) != len(picks):
        return "Each option can be picked once."
    if not low <= len(picks) <= high:
        if low == high:
            return f"Pick {low}."
        return f"Pick between {low} and {high}."
    return None


def _draw_choice(data: Dict[str, Any]) -> Dict[str, Any]:
    _show(data["question"], data.get("details"))
    options = data["options"]
    _list_options(options)
    low, high = data.get("min", 1), data.get("max", 1)
    picked = data.get("default") or []
    default = [i for i, option in enumerate(options) if option["id"] in picked]
    ask = "Number" if high == 1 else f"Numbers, comma-separated ({low} to {high})"
    picks = _pick(options, ask, default, low, high)
    return {"selected": [{"id": options[i]["id"]} for i in picks]}


def _draw_ontology(data: Dict[str, Any]) -> Dict[str, Any]:
    """The suggested terms first, then the whole tree, indented under each
    term's parent; one number picks any of them."""
    options = data["options"]
    known = {option["id"] for option in options}
    children: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for option in options:
        parent = option.get("parent") if option.get("parent") in known else None
        children.setdefault(parent, []).append(option)
    tree: List[Dict[str, Any]] = []

    def walk(parent: Optional[str], depth: int) -> None:
        for option in children.get(parent, []):
            tree.append({**option, "label": "  " * depth + _name(option)})
            walk(option["id"], depth + 1)

    walk(None, 0)
    suggested = [o for o in options if o["id"] in (data.get("suggested") or [])]
    shown = [{**o, "label": f"★ {_name(o)}"} for o in suggested] + tree
    if suggested:
        print(
            f"\n(★ = suggested; the {data.get('ontology') or 'ontology'} terms around them follow)"
        )
    default = [data["default"]] if data.get("default") else []
    reply = _draw_choice({**data, "options": shown, "default": default})
    return reply


def _draw_sequence(data: Dict[str, Any]) -> Dict[str, Any]:
    """The sequence's name and length, each region with its letters, then
    one number picks a region."""
    letters = data.get("sequence") or ""
    kind = (data.get("kind") or "sequence").upper()
    print(f"\n{data.get('name', '')}: {kind}, {len(letters)} letters")
    options = []
    for option in data["options"]:
        start, end = option.get("start"), option.get("end")
        part = letters[start - 1 : end] if start else ""
        if len(part) > 60:
            part = part[:57] + "..."
        where = f"{start}–{end}: {part}" if start else ""
        details = " · ".join(x for x in (where, option.get("details")) if x)
        options.append({**option, "details": details})
    default = [data["default"]] if data.get("default") else []
    return _draw_choice({**data, "options": options, "default": default})


def _draw_choice_list(data: Dict[str, Any]) -> Dict[str, Any]:
    """One prompt per available item; an option that needs one asks a reason."""
    _show(data["question"], data.get("details"))
    options = data["options"]
    _list_options(options)
    ids = [option["id"] for option in options]
    defaults = data.get("default") or {}
    answers: Dict[str, Any] = {}
    for item in data["items"]:
        name = _name(item) + (f" ({item['details']})" if item.get("details") else "")
        if item.get("unavailable"):
            print(f"  {name}: not answered here ({item['unavailable']})")
            continue
        before = defaults.get(item["id"]) or {}
        default = [ids.index(before["id"])] if before.get("id") in ids else []
        option = options[_pick(options, name, default)[0]]
        row: Dict[str, Any] = {"id": option["id"]}
        reason = option.get("reason")
        if reason:
            kept = None
            if before.get("id") == option["id"]:
                kept = (before.get("fields") or {}).get(reason["key"])
            text = _read_text(reason.get("question") or "Reason", kept)
            row["fields"] = {reason["key"]: text}
        answers[item["id"]] = row
    return {"answers": answers}


def _read_text(label: str, kept: Optional[str]) -> str:
    """Non-empty text; empty input keeps ``kept`` when there is one."""
    keep = f" [Enter = keep: {kept}]" if kept else ""
    while True:
        text = _read(f"  {label.rstrip(':?')}{keep}: ")
        if text or kept:
            return text or kept
        print("  An answer is needed.")


def _draw_plate(data: Dict[str, Any]) -> Dict[str, Any]:
    """One prompt per sample, named with its well; an option that needs one
    asks a reason. Samples that can't be assigned are listed, never asked."""
    _show(data["question"])
    if data.get("positions") == "list_order":
        print("  (No positions were given: the wells are in list order.)")
    options = data.get("options") or [{"id": c} for c in data.get("choices", [])]
    ids = [option["id"] for option in options]
    _list_options(options)
    assignments: Dict[str, Any] = {}
    reasons: Dict[str, Any] = {}
    for well in data["wells"]:
        place = f"{chr(ord('A') + well['row'])}{well['col'] + 1}"
        label = well.get("label") or ""
        who = label if well["sample_id"] in label else f"{well['sample_id']} {label}"
        name = f"{place} {who.strip()}"
        if well.get("excluded"):
            print(f"  Not assignable: {name} — {well['excluded']}")
            continue
        details = (well.get("details") or {}).items()
        shown = "; ".join(v if k == "note" else f"{k}: {v}" for k, v in details)
        if well.get("warning"):
            shown = "; ".join(filter(None, [shown, f"warning: {well['warning']}"]))
        if shown:
            name += f" ({shown})"
        condition = well.get("condition")
        default = [ids.index(condition)] if condition in ids else []
        option = options[_pick(options, name, default)[0]]
        assignments[well["sample_id"]] = option["id"]
        reason = option.get("reason")
        if reason:
            kept = well.get("reason") if condition == option["id"] else None
            reasons[well["sample_id"]] = _read_text(
                reason.get("question") or "Reason", kept
            )
    reply: Dict[str, Any] = {"assignments": assignments}
    if reasons:
        reply["reasons"] = reasons
    return reply


def _draw_slider(data: Dict[str, Any]) -> Dict[str, Any]:
    """A number within the slider's bounds; asks again until it is valid."""
    from lobster.services.interaction.component_schemas import plain_number

    low, high, default = data["min"], data["max"], data.get("default")
    # The adapter sends a step only for whole-number fields.
    whole = isinstance(data.get("step"), int)
    unit = f" {data['unit']}" if data.get("unit") else ""
    _show(data.get("label") or "")
    kind = "A whole number" if whole else "A number"
    print(f"  {kind} from {plain_number(low)} to {plain_number(high)}{unit}")
    keep = f" [Enter = {plain_number(default)}]" if default is not None else ""
    while True:
        text = _read(f"Value{keep}: ")
        if not text and default is not None:
            return {"value": default}
        try:
            value = float(text)
        except ValueError:
            print("Enter a number.")
            continue
        if whole and not value.is_integer():
            print("Enter a whole number.")
        elif not low <= value <= high:
            print(f"Enter a value from {plain_number(low)} to {plain_number(high)}.")
        else:
            return {"value": int(value) if whole else value}


def _draw_text(data: Dict[str, Any]) -> Dict[str, Any]:
    """A line of text; empty input keeps the default when there is one."""
    _show(data.get("question") or "")
    keep = data.get("placeholder")
    while True:
        text = _read(f"[{keep}]: " if keep else "> ")
        if text or keep:
            return {"answer": text}
        print("An answer is needed.")


def _draw_dashboard(data: Dict[str, Any]) -> Dict[str, Any]:
    """Every section in order, then a summary to confirm.

    y or yes sends; n, no or Enter decides later; anything else asks again,
    so a typo never throws the answers away.
    """
    print(f"\n{data['question']}")
    for problem in data.get("problems") or []:
        print(f"⚠ {problem}")
    sections = data["sections"]
    replies: Dict[str, Any] = {}
    for number, section in enumerate(sections, 1):
        print(f"\n— Part {number} of {len(sections)} —")
        for problem in section.get("problems") or []:
            print(f"⚠ {problem}")
        action, reply = _render(section["component"], section["data"])
        if action != "submit":
            raise _Stop(action)
        replies[section["key"]] = reply
    print("\nYour answers:")
    for section in sections:
        reply = replies[section["key"]]
        summary = _summary(section["component"], section["data"], reply)
        print(f"  {section['title']}: {summary}")
    while True:
        said = _read("Confirm all? [y/N]: ").lower()
        if said in ("y", "yes"):
            return {"answers": replies}
        if said in ("", "n", "no"):
            raise _Stop("defer")
        print("Type y to send these answers, or n to decide later.")


def _summary(component: str, data: Dict[str, Any], reply: Dict[str, Any]) -> str:
    """One line for a section's reply, for the confirmation summary."""
    from lobster.services.interaction.component_schemas import plain_number

    if component in ("choice", "ontology_browser", "sequence_view"):
        names = {option["id"]: _name(option) for option in data["options"]}
        return ", ".join(names[pick["id"]] for pick in reply["selected"]) or "none"
    if component == "choice_list":
        rows = []
        for item_id, row in reply["answers"].items():
            fields = "".join(f" ({k}: {v})" for k, v in row.get("fields", {}).items())
            rows.append(f"{item_id} {row['id']}{fields}")
        return "; ".join(rows)
    if component == "plate_map":
        groups = []
        reasons = reply.get("reasons") or {}
        options = data.get("options") or [{"id": c} for c in data.get("choices", [])]
        for option in options:
            ids = [
                i + (f" ({reasons[i]})" if reasons.get(i) else "")
                for i, picked in reply["assignments"].items()
                if picked == option["id"]
            ]
            if ids:
                groups.append(f"{_name(option)}: {', '.join(ids)}")
        return "; ".join(groups)
    if component == "threshold_slider":
        return plain_number(reply["value"])
    return reply.get("answer") or "(kept)"


_DRAW: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "choice": _draw_choice,
    "choice_list": _draw_choice_list,
    "plate_map": _draw_plate,
    "ontology_browser": _draw_ontology,
    "sequence_view": _draw_sequence,
    "threshold_slider": _draw_slider,
    "text_input": _draw_text,
    "dashboard": _draw_dashboard,
}
