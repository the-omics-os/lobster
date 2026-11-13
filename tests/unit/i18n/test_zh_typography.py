"""Simplified-Chinese catalog typography and terminology."""

import json
import re
from pathlib import Path


import lobster.i18n as i18n

CATALOG_DIR = Path(i18n.__file__).parent
CJK = "[　-〿一-鿿＀-￯]"
# Rich markup tags such as [cyan], [/], [bold {PRIMARY_ORANGE}]; not "[{default}]".
MARKUP = re.compile(r"\[/?(?:[a-zA-Z][^\[\]]*)?\]")
SPACE_BETWEEN_CJK = re.compile(CJK + " +" + CJK)
SPACE_AFTER_FULLWIDTH_PUNCT = re.compile(r"[：。] +(?=[一-鿿])")


def _values():
    for name in ("phrases.zh.json", "zh.json"):
        catalog = json.loads((CATALOG_DIR / name).read_text(encoding="utf-8"))
        for key, value in catalog.items():
            yield name, key, value


def test_no_ascii_space_between_chinese_characters():
    offenders = [
        (name, key)
        for name, key, value in _values()
        # Column-aligned menu rows keep the padding of their English source.
        if "     " not in key and SPACE_BETWEEN_CJK.search(MARKUP.sub("", value))
    ]
    assert offenders == []


def test_no_ascii_space_after_full_width_punctuation():
    offenders = [
        (name, key)
        for name, key, value in _values()
        if "     " not in key
        and SPACE_AFTER_FULLWIDTH_PUNCT.search(MARKUP.sub("", value))
    ]
    assert offenders == []


def test_tier_is_never_translated_as_dang():
    english = json.loads((CATALOG_DIR / "en.json").read_text(encoding="utf-8"))
    offenders = [
        (name, key)
        for name, key, value in _values()
        if "tier" in english.get(key, key).lower() and "档" in value
    ]
    assert offenders == []
