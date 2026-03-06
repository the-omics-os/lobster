"""Map natural-language questions to HITL UI components.

The mapper uses rule-based heuristics to select the appropriate component.
A future version may use an LLM with structured output for complex cases.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from lobster.services.interaction.component_schemas import (
    COMPONENT_SCHEMAS,
    ComponentSelection,
)

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
    question: str, context: Optional[Dict[str, Any]] = None
) -> ComponentSelection:
    """Map a question + context to a ComponentSelection.

    Uses rule-based heuristics:
    1. If context has "options" list -> select
    2. If context has "clusters" list -> cell_type_selector
    3. If context has threshold-related keys -> threshold_slider
    4. If question matches confirm patterns -> confirm
    5. Fallback -> text_input
    """
    ctx = context or {}

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

    # Fallback: text_input.
    placeholder = ctx.get("placeholder", "")
    return ComponentSelection(
        component="text_input",
        data={"question": question, "placeholder": placeholder},
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
