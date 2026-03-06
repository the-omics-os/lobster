"""Classic CLI fallback renderers for HITL interrupt components.

Used when the Go TUI is not available. Renders interrupt components as
simple terminal prompts using Python's built-in input().
"""

from __future__ import annotations

from typing import Any, Dict


def handle_interrupt_classic(interrupt_data: Dict[str, Any]) -> Dict[str, Any]:
    """Render an interrupt as a terminal prompt and collect the response.

    Args:
        interrupt_data: The interrupt payload from ComponentMapper output
                        (ComponentSelection.model_dump()).

    Returns:
        User's response as a dict matching the component's output_schema.
    """
    component = interrupt_data.get("component", "text_input")
    fallback = interrupt_data.get("fallback_prompt", "Please provide input:")
    data = interrupt_data.get("data", {})

    if component == "confirm":
        return _handle_confirm(fallback, data)
    elif component == "select":
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
