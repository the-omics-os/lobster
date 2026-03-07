"""HITL component schemas for interrupt-driven user interaction.

Defines the component registry and selection model used by the ComponentMapper
to translate natural-language questions into typed UI components.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ComponentSelection(BaseModel):
    """Result of mapping a question to a UI component."""

    component: str = Field(
        description="Component type key from COMPONENT_SCHEMAS"
    )
    mode: str = Field(
        default="overlay",
        description="Rendering mode: 'inline', 'overlay', or 'fullscreen'",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Component-specific typed payload",
    )
    fallback_prompt: str = Field(
        description="Plain-text prompt for classic CLI or unknown components"
    )

    @model_validator(mode="after")
    def _validate_component_and_fallback(self) -> "ComponentSelection":
        """Ensure component is valid and fallback_prompt is never empty."""
        # Validate component name against registry.
        if self.component not in COMPONENT_SCHEMAS:
            self.component = "text_input"
            # Ensure data has question key for text_input.
            if "question" not in self.data:
                self.data["question"] = self.fallback_prompt or "Please respond"

        # Ensure fallback_prompt is never empty/whitespace.
        if not self.fallback_prompt or not self.fallback_prompt.strip():
            self.fallback_prompt = self.data.get("question", "Please respond")

        return self


# Registry of available HITL components.
# Each entry describes input/output schemas so the mapper LLM can choose.
COMPONENT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "confirm": {
        "description": "Yes/no confirmation dialog",
        "input_schema": {"question": "str", "default": "bool"},
        "output_schema": {"confirmed": "bool"},
    },
    "select": {
        "description": "Single choice from a list of options",
        "input_schema": {"question": "str", "options": "list[str]"},
        "output_schema": {"selected": "str", "index": "int"},
    },
    "text_input": {
        "description": "Free-text answer (fallback for any question)",
        "input_schema": {"question": "str", "placeholder": "str"},
        "output_schema": {"answer": "str"},
    },
    "cell_type_selector": {
        "description": (
            "Assign cell type labels to single-cell clusters "
            "with marker gene context"
        ),
        "input_schema": {
            "clusters": "[{id: int, size: int, markers: list[str]}]"
        },
        "output_schema": {"assignments": "dict[str, str]"},
    },
    "threshold_slider": {
        "description": (
            "Adjust a numeric threshold with live preview of affected items"
        ),
        "input_schema": {
            "label": "str",
            "min": "float",
            "max": "float",
            "default": "float",
            "unit": "str",
        },
        "output_schema": {"value": "float"},
    },
}
