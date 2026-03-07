"""Map natural-language questions to HITL UI components.

Hybrid approach: rule-based fast path for unambiguous structural context,
LLM structured output for ambiguous/novel question patterns.
"""

from __future__ import annotations

import json as _json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, Optional

from lobster.services.interaction.component_schemas import (
    COMPONENT_SCHEMAS,
    ComponentSelection,
)

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
) -> ComponentSelection:
    """Map a question + context to a ComponentSelection.

    Hybrid approach:
    1. Rule-based fast path for unambiguous structural context
    2. LLM structured output for ambiguous/novel questions (when llm provided)
    3. Rule-based fallback when no LLM available

    Args:
        question: Natural language question for the user.
        context: Structural context (options, clusters, thresholds, etc.).
        llm: Optional LLM for classifying ambiguous questions.
    """
    ctx = context or {}

    # --- Rule-based fast path (unambiguous structural context) ---

    # Explicit options list -> select component.
    options = ctx.get("options")
    if isinstance(options, list) and len(options) > 0:
        return ComponentSelection(
            component="select",
            data={"question": question, "options": options},
            fallback_prompt=_build_select_fallback(question, options),
        )

    # Cluster data -> cell_type_selector.
    clusters = ctx.get("clusters")
    if isinstance(clusters, list) and len(clusters) > 0:
        return ComponentSelection(
            component="cell_type_selector",
            mode="overlay",
            data={"clusters": clusters},
            fallback_prompt=_build_cluster_fallback(question, clusters),
        )

    # Threshold/numeric range context -> threshold_slider.
    if _has_threshold_context(ctx) or _THRESHOLD_PATTERNS.search(question):
        return _build_threshold_selection(question, ctx)

    # Confirmation pattern -> confirm.
    if _CONFIRM_PATTERNS.search(question):
        default = ctx.get("default", False)
        return ComponentSelection(
            component="confirm",
            data={"question": question, "default": bool(default)},
            fallback_prompt=f"{question} [y/N]",
        )

    # --- LLM path (ambiguous questions) ---
    if llm is not None:
        return _llm_select_component(llm, question, ctx)

    # --- Fallback: text_input ---
    placeholder = ctx.get("placeholder", "")
    return ComponentSelection(
        component="text_input",
        data={"question": question, "placeholder": placeholder},
        fallback_prompt=question,
    )


def _llm_select_component(
    llm: "BaseChatModel",
    question: str,
    context: Dict[str, Any],
) -> ComponentSelection:
    """Use LLM structured output to select the best component.

    Builds a prompt listing available components and their schemas,
    then calls the LLM with structured output to get a ComponentSelection.
    Falls back to text_input on any error.
    """
    try:
        # Build component catalog for the prompt.
        component_lines = []
        for name, schema in COMPONENT_SCHEMAS.items():
            desc = schema.get("description", "")
            inputs = _json.dumps(schema.get("input_schema", {}))
            component_lines.append(f"- {name}: {desc} (inputs: {inputs})")
        catalog = "\n".join(component_lines)

        ctx_str = _json.dumps(context) if context else "{}"

        prompt = (
            "Select the best UI component for this user question.\n\n"
            f"Available components:\n{catalog}\n\n"
            f"User question: {question}\n"
            f"Context: {ctx_str}\n\n"
            "Return the component name, rendering mode, data payload, "
            "and a plain-text fallback prompt."
        )

        structured_llm = llm.with_structured_output(ComponentSelection)
        result = structured_llm.invoke(prompt)
        return result  # Pydantic validator handles invalid component names

    except Exception:
        logger.debug(
            "LLM component selection failed, falling back to text_input",
            exc_info=True,
        )
        return ComponentSelection(
            component="text_input",
            data={"question": question},
            fallback_prompt=question,
        )


def _has_threshold_context(ctx: Dict[str, Any]) -> bool:
    """Check if context contains threshold-related keys.

    Requires at least one of min/max/threshold/cutoff — "default" alone
    is not enough since many components use it.
    """
    strong_keys = {"min", "max", "threshold", "cutoff"}
    return bool(strong_keys & set(ctx.keys()))


def _build_threshold_selection(
    question: str, ctx: Dict[str, Any]
) -> ComponentSelection:
    """Build a threshold_slider selection from context."""
    data = {
        "label": question,
        "min": float(ctx.get("min", 0.0)),
        "max": float(ctx.get("max", 1.0)),
        "default": float(ctx.get("default", ctx.get("threshold", 0.5))),
        "unit": ctx.get("unit", ""),
    }
    unit_str = f" {data['unit']}" if data["unit"] else ""
    fallback = (
        f"{question} (range: {data['min']}-{data['max']}{unit_str}, "
        f"default: {data['default']}{unit_str})"
    )
    return ComponentSelection(
        component="threshold_slider",
        data=data,
        fallback_prompt=fallback,
    )


def _build_select_fallback(question: str, options: list) -> str:
    """Build a numbered fallback prompt for select."""
    lines = [question]
    for i, opt in enumerate(options, 1):
        lines.append(f"  {i}. {opt}")
    lines.append("Enter number: ")
    return "\n".join(lines)


def _build_cluster_fallback(question: str, clusters: list) -> str:
    """Build a text fallback for cluster annotation."""
    lines = [question]
    for c in clusters:
        cid = c.get("id", "?")
        size = c.get("size", "?")
        markers = ", ".join(c.get("markers", [])[:5])
        lines.append(f"  Cluster {cid} ({size} cells): {markers}")
    lines.append("Enter assignments as 'cluster_id=cell_type' per line:")
    return "\n".join(lines)
