"""Fixtures for the decision-components end-to-end tests (PHASE 5).

Run with::

    cd lobster-tui && go build -o lobster-tui ./cmd/lobster-tui && cd ..
    uv run pytest tests/integration/decision_components -m e2e_tui -q --no-cov

* The ``e2e_tui`` marker is registered here (not in pyproject.toml).
* The tests are opt-in: they are skipped unless ``-m`` selects ``e2e_tui``
  (or ``LOBSTER_E2E_TUI=1``), because they start real processes.
* Tests that need the binary are skipped, with the reason, when it is not
  built (``LOBSTER_TUI_BINARY`` or ``lobster-tui/lobster-tui``).
* Every session fixture closes the bridge, kills the binary's process group
  and reaps it, pass or fail.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterator, List

import pytest

from tests.integration.decision_components._harness import (
    TuiSession,
    binary_path,
    make_client,
    make_dataset,
)

_HERE = Path(__file__).resolve().parent


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e_tui: decision components end to end on the real lobster-tui binary "
        "under a pseudo-terminal (opt-in: -m e2e_tui)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: List[Any]) -> None:
    selected = "e2e_tui" in (config.getoption("markexpr") or "") or os.environ.get(
        "LOBSTER_E2E_TUI", ""
    ).lower() in {"1", "true", "yes"}
    skip_opt_in = pytest.mark.skip(
        reason="opt-in end-to-end TUI test: run with -m e2e_tui (or LOBSTER_E2E_TUI=1)"
    )
    for item in items:
        if _HERE not in Path(str(item.fspath)).resolve().parents:
            continue
        item.add_marker(pytest.mark.e2e_tui)
        if not selected:
            item.add_marker(skip_opt_in)


@pytest.fixture
def tui_binary() -> Path:
    path = binary_path()
    if path is None:
        wanted = os.environ.get("LOBSTER_TUI_BINARY") or "lobster-tui/lobster-tui"
        pytest.skip(
            f"lobster-tui binary not found at {wanted}; build it with "
            "`cd lobster-tui && go build -o lobster-tui ./cmd/lobster-tui`"
        )
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture(params=["review8", "review72"])
def any_dataset(request: pytest.FixtureRequest, workspace: Path):
    return make_dataset(request.param, workspace)


@pytest.fixture
def review8(workspace: Path):
    return make_dataset("review8", workspace)


@pytest.fixture
def review72(workspace: Path):
    return make_dataset("review72", workspace)


@pytest.fixture
def start_tui(tui_binary: Path, workspace: Path) -> Iterator[Callable[..., TuiSession]]:
    """start_tui(steps, rows=40, cols=120, extra_tools=()) -> a started TuiSession."""
    sessions: List[TuiSession] = []

    def start(steps, rows: int = 40, cols: int = 120, extra_tools=()) -> TuiSession:
        client = make_client(workspace, steps, extra_tools=extra_tools)
        session = TuiSession(tui_binary, client, workspace, rows=rows, cols=cols)
        sessions.append(session)
        return session.start()

    yield start
    for session in sessions:
        session.close()
