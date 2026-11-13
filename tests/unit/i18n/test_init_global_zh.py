"""Chinese text on the `lobster init --global` and Ollama setup screens."""

import lobster.i18n as i18n
from lobster.config.provider_setup import get_ollama_install_instructions

GLOBAL_BANNER = (
    "[bold white]🦞 Welcome to Lobster AI![/bold white]\n\n"
    "Let's set up your [cyan]global[/cyan] provider defaults.\n"
)


def test_ollama_windows_install_hint_is_translated():
    i18n.resolve_lang(["lobster", "--lang", "zh"])
    hint = get_ollama_install_instructions()["windows"]
    assert i18n.h(hint) != hint
    assert "https://ollama.com/download" in i18n.h(hint)


def test_global_banner_has_no_spaces_inside_the_chinese_sentence():
    i18n.resolve_lang(["lobster", "--lang", "zh"])
    key = next(k for k in i18n._phrases() if k.startswith(GLOBAL_BANNER))
    assert "我们来设置你的[cyan]全局[/cyan]模型服务默认值" in i18n.h(key)
