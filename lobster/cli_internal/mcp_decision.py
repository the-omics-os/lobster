"""The question a Lobster run asks a host, and the answers that fit it.

Pure functions, stdlib only. ``view`` turns the run API's decision summary into
what ``lobster_status`` shows. ``check_answer`` is a light shape check per
component kind: the engine accepts any JSON value, so this server keeps obvious
garbage out of the record and hands the answer on in the structured form the
TUI components return (``component_schemas.COMPONENT_SCHEMAS``, the Go
components in ``lobster-tui/internal/biocomp``), so records look alike on every
surface.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict

# What the live status shows of a question. A long question or option list must
# not push decision_id and the answer shapes out of a host's reader.
SHOWN_MAX_CHARS = 2000
MAX_OPTIONS = 50
OPTION_MAX_CHARS = 300
LISTED_IN_REFUSAL = 20
CUT = "... (cut)"
# The answers of a turn as its record and status list them.
MAX_RECORDED = 50
RECORDED_QUESTION_MAX_CHARS = 500
RECORDED_COMMENT_MAX_CHARS = 1000
RECORDED_NOTE = "Answers were reported by the host; Lobster cannot check who gave them."

# What each component kind accepts; the first entry is the form that is recorded.
ANSWER_SHAPES: Dict[str, list] = {
    "confirm": ['{"confirmed": true or false}', "true or false", '"yes" or "no"'],
    # Never a number: a position and the "1. 2. 3." of a prompt count from
    # different ends, and a wrong pick would be recorded as the host's choice.
    "select": ['{"selected": <option text>}', "the option text"],
    "text_input": ['{"answer": <text>}', "non-empty text"],
    "threshold_slider": [
        '{"value": <number>}',
        "a number from min to max when the question gives them",
    ],
    "cell_type_selector": [
        '{"assignments": {"<cluster id>": "<cell type>"}}',
        '{"<cluster id>": "<cell type>"}',
    ],
}
ANY_JSON = ["any string, number, boolean or object"]
_YES, _NO = {"yes", "y", "true"}, {"no", "n", "false"}


class BadAnswer(ValueError):
    """An answer that does not fit the question; the message says why."""


def cut(text: str, limit: int) -> str:
    """*text* cut to *limit* characters, the cut marked."""
    return text if len(text) <= limit else text[: limit - len(CUT)] + CUT


def bounded(value: Any, limit: int) -> Any:
    """*value* as is when its JSON is short, else its text cut at *limit*."""
    if isinstance(value, str):
        return cut(value, limit)
    text = json.dumps(value, default=str)
    return value if len(text) <= limit else cut(text, limit)


def view(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The ``decision`` object of a ``needs_decision`` status.

    *summary* is ``Decision.summary()`` (``lobster.core.runs``); only the first
    component is shown, the engine emits one per decision. Text is cut and at
    most ``MAX_OPTIONS`` options are shown, with ``options_truncated`` and
    ``options_total``; the full list stays under ``_options`` (never sent) so
    ``check_answer`` still accepts an option that was not shown.
    """
    component = (summary.get("components") or [{}])[0]
    data = component.get("data") if isinstance(component.get("data"), dict) else {}
    kind = component.get("component")
    shown: Dict[str, Any] = {
        "decision_id": summary["decision_id"],
        "request_version": summary["request_version"],
        "question": cut(str(summary.get("question", "")), SHOWN_MAX_CHARS),
        "kind": kind,
    }
    for key in ("options", "min", "max", "default"):
        if key in data:
            shown[key] = data[key]
    options = shown.get("options")
    if isinstance(options, list):
        shown["options"] = [bounded(o, OPTION_MAX_CHARS) for o in options[:MAX_OPTIONS]]
        if len(options) > MAX_OPTIONS:
            shown["options_truncated"] = True
            shown["options_total"] = len(options)
        if shown["options"] != options:
            shown["_options"] = options
    for key in ("min", "max", "default"):
        if key in shown:
            shown[key] = bounded(shown[key], SHOWN_MAX_CHARS)
    if kind != "select":  # a select's prompt only repeats the options, numbered
        prompt = component.get("fallback_prompt") or str(summary.get("question", ""))
        shown["fallback_prompt"] = cut(str(prompt), SHOWN_MAX_CHARS)
    shown["answer_shapes"] = list(ANSWER_SHAPES.get(kind, ANY_JSON))
    return shown


def recorded_entry(
    decision: Dict[str, Any],
    answer: Any,
    source: str,
    channel: str,
    comment: Any = None,
) -> Dict[str, Any]:
    """One answer as the record lists it: the question as it was shown (cut),
    the accepted *answer* in its recorded form, and *source* and *channel* as
    the host reported them. ``comment`` is there only when one was given."""
    entry: Dict[str, Any] = {
        "question": cut(str(decision.get("question", "")), RECORDED_QUESTION_MAX_CHARS),
        "answer": answer,
        "source": source,
        "channel": channel,
    }
    if comment is not None:
        entry["comment"] = cut(str(comment), RECORDED_COMMENT_MAX_CHARS)
    return entry


def public(decision: Dict[str, Any]) -> Dict[str, Any]:
    """*decision* as the host sees it: without the private ``_`` keys."""
    return {k: v for k, v in decision.items() if not k.startswith("_")}


def check_answer(decision: Dict[str, Any], answer: Any) -> Any:
    """*answer* in the form the component returns, or ``BadAnswer``.

    *decision* is a ``view``. A kind this server does not know passes the
    answer on as it came.
    """
    checks = {
        "confirm": _confirm,
        "select": _select,
        "text_input": _text,
        "threshold_slider": _threshold,
        "cell_type_selector": _assignments,
    }
    check = checks.get(decision.get("kind"))
    return answer if check is None else check(decision, answer)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _confirm(decision: Dict[str, Any], answer: Any) -> Dict[str, bool]:
    if isinstance(answer, dict) and set(answer) == {"confirmed"}:
        answer = answer["confirmed"]
    elif isinstance(answer, str) and answer.strip().lower() in _YES | _NO:
        answer = answer.strip().lower() in _YES
    if not isinstance(answer, bool):
        raise BadAnswer("answer must be true or false (or yes or no).")
    return {"confirmed": answer}


def _same(option: Any, wanted: Any) -> bool:
    """Equal and of the same type: ``True`` is not the option ``1``."""
    return type(option) is type(wanted) and option == wanted


def _select(decision: Dict[str, Any], answer: Any) -> Dict[str, Any]:
    """The option text, or ``{"selected": <option text>}``; never a number."""
    options = decision.get("_options", decision.get("options")) or []
    listed = options[:LISTED_IN_REFUSAL]
    more = (
        f" (first {LISTED_IN_REFUSAL} of {len(options)})" if listed != options else ""
    )

    def refuse(value: Any) -> BadAnswer:
        return BadAnswer(
            f"{cut(repr(value), 100)} is not one of the options "
            f"{cut(json.dumps(listed, default=str), 800)}{more}; answer with "
            "one option text."
        )

    if isinstance(answer, dict) and "selected" in answer:
        wanted = answer["selected"]
    elif isinstance(answer, str):
        wanted = answer
    else:
        raise refuse(answer)
    first = next((i for i, o in enumerate(options) if _same(o, wanted)), None)
    if first is None:
        raise refuse(wanted)
    if isinstance(answer, dict) and "index" in answer:
        index = answer["index"]  # the TUI form carries one; it must agree
        if not (
            type(index) is int
            and 0 <= index < len(options)
            and _same(options[index], wanted)
        ):
            raise BadAnswer(
                f"index {cut(repr(index), 40)} does not point at "
                f"{cut(repr(wanted), 100)} in the options; answer with one "
                "option text."
            )
    return {"selected": options[first], "index": first}


def _text(decision: Dict[str, Any], answer: Any) -> Dict[str, str]:
    if isinstance(answer, dict) and set(answer) == {"answer"}:
        answer = answer["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise BadAnswer("answer must be non-empty text.")
    return {"answer": answer}


def _threshold(decision: Dict[str, Any], answer: Any) -> Dict[str, float]:
    if isinstance(answer, dict) and set(answer) == {"value"}:
        answer = answer["value"]
    try:  # a huge int overflows float()
        value = float(answer) if _is_number(answer) else math.nan
    except OverflowError:
        value = math.nan
    if not math.isfinite(value):
        raise BadAnswer("answer must be a finite number.")
    low, high = decision.get("min"), decision.get("max")
    if _is_number(low) and _is_number(high) and not low <= value <= high:
        raise BadAnswer(f"{value} is outside the range: between {low} and {high}.")
    return {"value": value}


def _assignments(decision: Dict[str, Any], answer: Any) -> Dict[str, Dict[str, str]]:
    if isinstance(answer, dict) and set(answer) == {"assignments"}:
        answer = answer["assignments"]
    if (
        not isinstance(answer, dict)
        or not answer
        or not all(isinstance(v, str) and v.strip() for v in answer.values())
    ):
        raise BadAnswer(
            'answer must be {"assignments": {"<cluster id>": "<cell type>"}} '
            "with at least one cluster."
        )
    return {"assignments": {str(k): v for k, v in answer.items()}}
