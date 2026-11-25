"""Dependency-free local GUI for already-saved Plotly figures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lobster.core.figures.store import LocalFigureStore


def launch_editor(store: LocalFigureStore, ref: str) -> dict[str, Any]:
    from .server import launch_editor as launch

    return launch(store, ref)
