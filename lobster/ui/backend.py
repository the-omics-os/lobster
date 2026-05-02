"""UI backend selection and Go frontend discovery utilities."""

from __future__ import annotations

import importlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

VALID_UI_BACKENDS = {"auto", "go", "classic"}
DEFAULT_UI_BACKEND = "auto"


@dataclass(frozen=True)
class BackendSelection:
    """Resolved UI backend selection."""

    requested: str
    resolved: str
    go_binary: Optional[str] = None
    reason: Optional[str] = None


def normalize_ui_backend(value: Optional[str]) -> str:
    """Normalize and validate a UI backend value."""
    candidate = (value or "").strip().lower() or DEFAULT_UI_BACKEND
    if candidate not in VALID_UI_BACKENDS:
        valid = ", ".join(sorted(VALID_UI_BACKENDS))
        raise ValueError(f"Invalid UI backend '{value}'. Expected one of: {valid}")
    return candidate


def resolve_ui_backend(requested: Optional[str]) -> str:
    """Resolve backend from CLI value and environment."""
    if requested:
        return normalize_ui_backend(requested)
    env_value = os.environ.get("LOBSTER_UI_BACKEND")
    if env_value:
        return normalize_ui_backend(env_value)
    return DEFAULT_UI_BACKEND


def find_go_tui_binary(explicit_path: Optional[str] = None) -> Optional[str]:
    """Find lobster-tui binary in common locations."""
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    env_path = os.environ.get("LOBSTER_TUI_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    # Monorepo development location.
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repo_root / "lobster-tui" / "lobster-tui",
            repo_root / "lobster-tui" / "bin" / "lobster-tui",
            repo_root / "lobster-tui" / "dist" / "lobster-tui",
        ]
    )

    for path in candidates:
        if path.exists() and path.is_file() and os.access(path, os.X_OK):
            return str(path)

    which_bin = shutil.which("lobster-tui")
    if which_bin:
        return which_bin

    # Optional packaged binary.
    try:
        pkg = importlib.import_module("lobster_ai_tui")
        get_path = getattr(pkg, "get_binary_path", None)
        if callable(get_path):
            p = Path(get_path())
            if p.exists() and os.access(p, os.X_OK):
                return str(p)
    except Exception:
        pass

    return None


def select_ui_backend(
    requested: Optional[str],
    *,
    explicit_go_binary: Optional[str] = None,
) -> BackendSelection:
    """Resolve final backend and Go binary selection."""
    resolved_request = resolve_ui_backend(requested)

    if resolved_request == "classic":
        return BackendSelection(requested=resolved_request, resolved="classic")

    go_binary = find_go_tui_binary(explicit_go_binary)

    if resolved_request == "go":
        if not go_binary:
            return BackendSelection(
                requested=resolved_request,
                resolved="go",
                go_binary=None,
                reason="Go UI requested but lobster-tui binary was not found.",
            )
        return BackendSelection(
            requested=resolved_request,
            resolved="go",
            go_binary=go_binary,
        )

    # auto mode
    if go_binary:
        return BackendSelection(
            requested=resolved_request,
            resolved="go",
            go_binary=go_binary,
        )

    return BackendSelection(
        requested=resolved_request,
        resolved="classic",
        reason="Go UI unavailable; falling back to classic UI.",
    )
