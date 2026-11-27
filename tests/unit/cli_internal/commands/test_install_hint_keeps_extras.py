"""An install command in an error keeps its [extras] wherever it is shown.

Rich took "[anthropic,openai]" in `lobster-ai[anthropic,openai]>=1.3.1` for a style
tag and dropped it, so the advice uninstalled the provider that worked.
"""

import io
import logging
import sys
import types

import pytest
from rich.console import Console

from lobster.cli_internal.commands.light.config_commands import config_provider_switch
from lobster.cli_internal.commands.output_adapter import (
    ConsoleOutputAdapter,
    ProtocolOutputAdapter,
)

COMMAND = 'uv tool install "lobster-ai[anthropic,openai]>=1.3.1" --python 3.12'
HINT = "Or add one provider: uv pip install 'lobster-ai[openai]'"
REFUSED = {
    "success": False,
    "error": "Failed to switch provider: langchain-openai package not installed. "
    f"Install with: {COMMAND}",
    "hint": HINT,
}


class _Client:
    def switch_provider(self, name):
        return dict(REFUSED)


def _console():
    return Console(file=io.StringIO(), width=300, color_system=None)


def test_classic_chat_keeps_the_extras():
    console = _console()
    assert (
        config_provider_switch(_Client(), ConsoleOutputAdapter(console), "openai")
        is None
    )
    text = console.file.getvalue()
    assert COMMAND in text
    assert HINT in text


def test_the_go_tui_keeps_the_extras_without_a_backslash():
    sent = []
    adapter = ProtocolOutputAdapter(lambda kind, payload: sent.append(payload))
    config_provider_switch(_Client(), adapter, "openai")
    text = "".join(payload.get("content", "") for payload in sent)
    assert COMMAND in text
    assert HINT in text
    assert "\\" not in text


def test_a_logged_warning_keeps_the_extras(monkeypatch):
    from lobster.ui import console_manager

    handlers = []
    monkeypatch.setattr(
        console_manager.logging,
        "basicConfig",
        lambda **kwargs: handlers.extend(kwargs["handlers"]),
    )
    monkeypatch.setattr(console_manager, "_set_engine_log_level", lambda level: None)
    quiet = ("httpx", "httpcore", "urllib3", "docling_core.transforms")
    levels = {name: logging.getLogger(name).level for name in quiet}
    console = _console()
    try:
        console_manager.LobsterConsoleManager._setup_logging(
            types.SimpleNamespace(_console=console)
        )
    finally:
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
    record = logging.LogRecord(
        "lobster.docling",
        logging.WARNING,
        __file__,
        1,
        f"Install with: {COMMAND}",
        None,
        None,
    )
    handlers[0].emit(record)
    assert COMMAND in console.file.getvalue()


def test_ollama_cloud_names_its_install_command(monkeypatch):
    from lobster.cli_internal.startup_diagnostics import _extract_install_command
    from lobster.config.providers.ollama_cloud_provider import OllamaCloudProvider

    monkeypatch.setitem(sys.modules, "langchain_openai", None)
    with pytest.raises(ImportError) as raised:
        OllamaCloudProvider().check_dependencies()
    assert "[openai]" in (_extract_install_command(str(raised.value)) or "")


def test_an_escaped_tag_is_text_and_never_closes_a_tag():
    from lobster.cli_internal.commands.output_adapter import _strip_markup

    assert _strip_markup("[red]x \\[a,b] y[/]") == "x [a,b] y"
    assert _strip_markup("\\[exports|all]") == "[exports|all]"
    # Two backslashes are one literal backslash and the tag still closes.
    assert _strip_markup("[red]C:\\dir\\\\[/red]") == "C:\\dir\\"
    # escape() never writes this; Rich itself would drop the backslash here.
    assert _strip_markup("[dim]\\[--global][/dim]") == "\\[--global]"


def test_json_output_keeps_code_as_written():
    from lobster.cli_internal.commands.output_adapter import (
        JsonOutputAdapter,
        code_block,
    )

    adapter = JsonOutputAdapter()
    source = 're.compile(r"\\[(\\d+)\\]")  # [red]not markup[/red]'
    adapter.render_blocks([code_block(source)])
    assert adapter.blocks[-1]["code"] == source
