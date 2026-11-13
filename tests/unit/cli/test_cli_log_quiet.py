"""CLI host log-level contract: engine INFO noise stays off the CLI by default.

The engine's ``lobster`` namespace logger defaults to INFO (engine/cloud
default). The CLI host must lower it to its own resolved level, and an explicit
``LOBSTER_LOG_LEVEL`` must still win.
"""

import logging
import os
import re
import subprocess  # nosec B404 # test runs the CLI as a child process
import sys

import pytest

from lobster.ui import console_manager as cm


@pytest.fixture
def restore_logging():
    """Snapshot and restore logger/handler state the console manager mutates."""
    names = ["", "lobster", "httpx", "httpcore", "urllib3", "docling_core.transforms"]
    root = logging.getLogger()
    saved_levels = {n: logging.getLogger(n).level for n in names}
    saved_handlers = list(root.handlers)
    saved_handler_levels = {h: h.level for h in saved_handlers}
    yield
    for n, lvl in saved_levels.items():
        logging.getLogger(n).setLevel(lvl)
    root.handlers[:] = saved_handlers
    for h, lvl in saved_handler_levels.items():
        h.setLevel(lvl)


def _fresh_manager():
    return cm.LobsterConsoleManager()


def test_default_cli_level_silences_engine_info(monkeypatch, restore_logging):
    monkeypatch.delenv("LOBSTER_LOG_LEVEL", raising=False)
    _fresh_manager()._setup_logging()
    assert not logging.getLogger("lobster.core.component_registry").isEnabledFor(
        logging.INFO
    )
    assert logging.getLogger("lobster.core.component_registry").isEnabledFor(
        logging.WARNING
    )


def test_explicit_info_override_keeps_engine_info(monkeypatch, restore_logging):
    monkeypatch.setenv("LOBSTER_LOG_LEVEL", "INFO")
    _fresh_manager()._setup_logging()
    assert logging.getLogger("lobster.core.component_registry").isEnabledFor(
        logging.INFO
    )


def test_level_is_independent_of_namespace_configuration_order(
    monkeypatch, restore_logging
):
    """A first get_logger() after the CLI setLevel must not reset it to INFO."""
    import lobster.utils.logger as engine_logger

    monkeypatch.delenv("LOBSTER_LOG_LEVEL", raising=False)
    monkeypatch.setattr(engine_logger, "_namespace_configured", False)
    _fresh_manager()._setup_logging()
    engine_logger.get_logger("lobster.some_late_module")
    assert not logging.getLogger("lobster.some_late_module").isEnabledFor(logging.INFO)


def test_setup_logging_debug_raises_engine_logger(monkeypatch, restore_logging):
    monkeypatch.delenv("LOBSTER_LOG_LEVEL", raising=False)
    cm.setup_logging(logging.DEBUG)
    assert logging.getLogger("lobster").level == logging.DEBUG
    assert logging.getLogger("lobster.core.component_registry").isEnabledFor(
        logging.DEBUG
    )


def test_setup_logging_warning_lowers_engine_logger(monkeypatch, restore_logging):
    monkeypatch.delenv("LOBSTER_LOG_LEVEL", raising=False)
    logging.getLogger("lobster").setLevel(logging.INFO)
    cm.setup_logging(logging.WARNING)
    assert logging.getLogger("lobster").level == logging.WARNING


@pytest.mark.slow
@pytest.mark.no_auto_config
def test_status_subprocess_has_no_info_or_pkg_resources_noise(tmp_path):
    """Real `lobster status` entry: no engine INFO lines, no sgkit warning."""
    env = {k: v for k, v in os.environ.items() if k != "LOBSTER_LOG_LEVEL"}
    env["LOBSTER_WORKSPACE"] = str(tmp_path / "ws")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(  # nosec B603 # fixed argv, no shell
        [sys.executable, "-m", "lobster.cli", "status"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = result.stdout + result.stderr
    info_lines = [ln for ln in output.splitlines() if re.search(r"\bINFO\b", ln)]
    assert info_lines == [], output
    assert "pkg_resources is deprecated" not in output, output
