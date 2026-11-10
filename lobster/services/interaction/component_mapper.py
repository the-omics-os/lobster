"""Turn an ``ask_user`` question and context into decision fields.

Context that already lists ``fields`` is validated and used as is. Older
hint-style context ({"options": [...]}, {"clusters": [...]}, threshold keys)
is converted by rules. An ambiguous question is formatted by an optional LLM,
so a separate model can take this work off the supervisor.
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import math
import re
from typing import TYPE_CHECKING, Any, Dict, Optional

from lobster.core.exceptions import InferenceRefused, reraise_inference_refusal
from lobster.services.interaction.component_schemas import AskUserContext

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Patterns that strongly suggest a confirmation question.
_CONFIRM_PATTERNS = re.compile(
    r"\b(proceed|continue|confirm|approve|accept|do you want|should I|"
    r"would you like|yes or no|y/n)\b",
    re.IGNORECASE,
)

# Patterns suggesting a threshold/numeric input.
_THRESHOLD_PATTERNS = re.compile(
    r"\b(threshold|cutoff|p-value|fdr|log.?fc|fold.?change|"
    r"significance|alpha|min|max)\b",
    re.IGNORECASE,
)


def map_question(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    llm: Optional["BaseChatModel"] = None,
) -> AskUserContext:
    """Map a question + context to the fields to ask.

    1. ``context["fields"]`` is validated and returned (raises ValidationError
       when invalid, so the caller can report the problems).
    2. Rule-based conversion of hint-style context.
    3. LLM structured output for ambiguous questions (when llm provided).
    4. A single free-text field.

    Args:
        question: Natural language question for the user.
        context: Fields, or older structural hints (options, clusters, ...).
        llm: Optional LLM for formatting ambiguous questions.
    """
    ctx = context or {}

    if "fields" in ctx:
        return AskUserContext.model_validate(ctx)
    asked = _from_hints(question, ctx, llm)
    # Keys that mean the same on any context are kept, and judged like any
    # other: never dropped silently (a supervisor that attached data_files
    # must get the fingerprint, or invalid_context).
    kept = {key: ctx[key] for key in _CONTEXT_KEYS if key in ctx}
    if not kept:
        return asked
    return AskUserContext.model_validate(
        {**asked.model_dump(exclude_none=True), **kept}
    )


# Context keys valid whatever builds the fields.
_CONTEXT_KEYS = ("view", "checks", "data_files")


def _from_hints(
    question: str, ctx: Dict[str, Any], llm: Optional["BaseChatModel"]
) -> AskUserContext:
    """The fields for hint-style context (no "fields" key)."""
    # --- Rule-based fast path (unambiguous structural context) ---

    # Explicit options list -> choice. Dict options keep their id.
    options = ctx.get("options")
    if isinstance(options, list) and len(options) > 0:
        return _single(
            {"key": "answer", "type": "choice", "options": _to_items(options)}
        )

    # Cluster data -> one text answer (the cell type) per cluster.
    clusters = ctx.get("clusters")
    if isinstance(clusters, list) and len(clusters) > 0:
        return _single(
            {"key": "cell_type", "type": "text", "for_each": _cluster_items(clusters)}
        )

    # Threshold/numeric range context -> number on a slider.
    if _has_threshold_context(ctx) or _THRESHOLD_PATTERNS.search(question):
        return _single(_threshold_field(ctx))

    # Confirmation pattern -> yes/no choice.
    if _CONFIRM_PATTERNS.search(question):
        field: Dict[str, Any] = {
            "key": "answer",
            "type": "choice",
            "options": ["yes", "no"],
        }
        if "default" in ctx:
            field["default"] = {"id": "yes" if ctx["default"] else "no"}
        return _single(field)

    # --- LLM path (ambiguous questions) ---
    if llm is not None:
        return _llm_format_question(llm, question, ctx)

    # --- Fallback: free text ---
    return _text_only(question, ctx.get("placeholder"))


def _llm_format_question(
    llm: "BaseChatModel",
    question: str,
    context: Dict[str, Any],
) -> AskUserContext:
    """Use LLM structured output to format the question as fields.

    Falls back to a single free-text field on any error.
    """
    try:
        ctx_str = _json.dumps(context) if context else "{}"
        prompt = (
            "Format this question for the user as decision fields.\n\n"
            "Field types: choice (pick options by stable id), number (min/max are "
            "value bounds), text (only when the answer can't be listed or "
            "counted), record (several fields answered together).\n\n"
            f"User question: {question}\n"
            f"Context: {ctx_str}\n"
        )
        structured_llm = llm.with_structured_output(AskUserContext)
        # nostream: this runs inside ask_user's build task, and the chat stream
        # would show a call made there as a sub-agent working.
        return structured_llm.invoke(prompt, config={"tags": ["nostream"]})

    except InferenceRefused:
        raise
    except Exception as error:
        reraise_inference_refusal(error)
        logger.debug(
            "LLM question formatting failed (error_type=%s, response_length=0, response_sha256=%s)",
            type(error).__name__,
            hashlib.sha256(b"").hexdigest(),
        )
    # Do not let a caller-side fallback failure chain model response values.
    return _text_only(question)


def _single(field: Dict[str, Any]) -> AskUserContext:
    return AskUserContext(fields=[field])


def _text_only(question: str, placeholder: Any = None) -> AskUserContext:
    """One free-text field; a hint's placeholder (an example answer) is shown
    as its details, never filled in as the answer."""
    field = {"key": "answer", "type": "text", "question": question}
    if placeholder:
        field["details"] = str(placeholder)
    return AskUserContext(fields=[field])


def _to_items(options: list) -> list:
    """Hint options as items. Dicts keep their id, details and unavailable
    reason (an ineligible option must stay unpickable); plain values are their
    own id."""
    items = []
    for opt in options:
        if isinstance(opt, dict):
            label = str(opt.get("label", opt.get("name", opt.get("id", ""))))
            item = {"id": str(opt.get("id", label)), "label": label}
            for key in ("details", "unavailable"):
                if opt.get(key):
                    item[key] = str(opt[key])
            items.append(item)
        else:
            items.append(str(opt))
    return items


def _cluster_items(clusters: list) -> list:
    """One for_each item per cluster; size and top markers become details."""
    items = []
    for c in clusters:
        item: Dict[str, Any] = {
            "id": str(c.get("id", "?")),
            "label": f"Cluster {c.get('id', '?')}",
        }
        parts = [f"{c['size']} cells" if "size" in c else ""]
        parts.append(", ".join(c.get("markers", [])[:5]))
        if any(parts):
            item["details"] = " · ".join(part for part in parts if part)
        items.append(item)
    return items


def _has_threshold_context(ctx: Dict[str, Any]) -> bool:
    """Check if context contains threshold-related keys.

    Requires at least one of min/max/threshold/cutoff — "default" alone
    is not enough since many components use it.
    """
    strong_keys = {"min", "max", "threshold", "cutoff"}
    return bool(strong_keys & set(ctx.keys()))


def _threshold_field(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """A number field: on a slider only when both bounds are given.

    Bounds are never invented: a 0-1 range fits a fraction, not a
    concentration or a read count. Without both bounds the user types the
    number. The default is kept when it is inside the bounds that were given.
    """
    field: Dict[str, Any] = {"key": "value", "type": "number"}
    low, high = _number(ctx.get("min")), _number(ctx.get("max"))
    if low is not None:
        field["min"] = low
    if high is not None:
        field["max"] = high
    if "min" in field and "max" in field:
        field["view"] = "slider"
    default = _number(ctx.get("default", ctx.get("threshold", ctx.get("cutoff"))))
    if default is not None and field.get("min", default) <= default <= field.get(
        "max", default
    ):
        field["default"] = default
    if ctx.get("unit"):
        field["unit"] = str(ctx["unit"])
    return field


def _number(value: Any) -> Optional[float]:
    """A finite number from a hint value (numbers, or numbers written as
    text such as "0.05"); None for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None
