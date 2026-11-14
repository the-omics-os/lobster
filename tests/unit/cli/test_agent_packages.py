"""The expert-package catalog lives in a light module the init wizard re-exports."""

import json
import subprocess
import sys
from pathlib import Path

from lobster.cli_internal import agent_packages

PHRASES = Path(__file__).resolve().parents[3] / "lobster" / "i18n" / "phrases.zh.json"


def test_the_catalog_is_ten_five_tuples_of_plain_strings():
    catalog = agent_packages.AVAILABLE_AGENT_PACKAGES
    assert len(catalog) == 10
    for name, description, agents, published, experimental in catalog:
        assert name.startswith("lobster-") and isinstance(description, str)
        assert agents and all(isinstance(a, str) for a in agents)
        assert isinstance(published, bool) and isinstance(experimental, bool)


def test_every_description_has_a_chinese_phrase():
    phrases = json.loads(PHRASES.read_text(encoding="utf-8"))
    missing = [
        d for _, d, *_ in agent_packages.AVAILABLE_AGENT_PACKAGES if d not in phrases
    ]
    assert missing == []


def test_init_reexports_the_same_packages_translated_for_display():
    from lobster.cli_internal.commands.heavy import init_commands

    shown = init_commands.AVAILABLE_AGENT_PACKAGES
    plain = agent_packages.AVAILABLE_AGENT_PACKAGES
    assert [(n, a, p, e) for n, _, a, p, e in shown] == [
        (n, a, p, e) for n, _, a, p, e in plain
    ]


def test_the_module_imports_without_the_engine():
    code = (
        "import sys, lobster.cli_internal.agent_packages;"
        "print([m for m in sys.modules if m.split('.')[0] in "
        "('langchain','langgraph','rich','typer','pydantic')])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    ).stdout
    assert out.strip() == "[]"
