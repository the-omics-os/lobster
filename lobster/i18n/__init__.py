"""Local UI language catalogs.

English (``en.json``) is the complete catalog. Other files are overlays.
A missing key or an unknown language returns English. This module uses only
the standard library so ``lobster/cli.py`` can load it before Typer builds help.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, MutableSequence, Optional

_DIR = Path(__file__).resolve().parent
_LANG = "en"
_CACHE: dict[str, dict[str, str]] = {}

# Easy-Mandarin overlay. Aliases all select ``zh``.
_ALIASES = {
    "en": "en",
    "english": "en",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "cn": "zh",
    "chinese": "zh",
    "mandarin": "zh",
}


def catalog_dir() -> Path:
    return _DIR


def current_lang() -> str:
    return _LANG


def canonicalize(lang: Optional[str]) -> str:
    if not lang:
        return "en"
    key = lang.strip().lower().replace("_", "-")
    return _ALIASES.get(key, key)


def _load(code: str) -> dict[str, str]:
    if code in _CACHE:
        return _CACHE[code]
    path = _DIR / f"{code}.json"
    if not path.is_file():
        _CACHE[code] = {}
        return _CACHE[code]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    _CACHE[code] = {str(k): str(v) for k, v in data.items()}
    return _CACHE[code]


def known_overlay(code: str) -> bool:
    return bool(re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]+)*", code)) and (
        code == "en" or (_DIR / f"{code}.json").is_file()
    )


def consume_lang_flag(argv: Optional[MutableSequence[str]]) -> Optional[str]:
    """Remove ``--lang`` from argv. The last occurrence wins."""
    if not argv or len(argv) < 2:
        return None
    found: Optional[str] = None
    kept = [argv[0]]
    i = 1
    while i < len(argv):
        item = argv[i]
        if item == "--":
            kept.extend(argv[i:])
            break
        if item == "--lang" and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            found = argv[i + 1]
            i += 2
            continue
        if item.startswith("--lang="):
            found = item.split("=", 1)[1]
            i += 1
            continue
        kept.append(item)
        i += 1
    if found is not None:
        argv[:] = kept
    return found


def read_workspace_lang(workspace: Optional[Path] = None) -> Optional[str]:
    """Read ``lang`` from provider_config.json. Does not open config.toml."""
    root = workspace if workspace is not None else Path.cwd()
    path = root / "provider_config.json"
    if not path.is_file():
        path = root / ".lobster_workspace" / "provider_config.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("lang")
    if isinstance(value, str) and value.strip():
        return value
    return None


def resolve_lang(
    argv: Optional[MutableSequence[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
    workspace: Optional[Path] = None,
) -> str:
    """Pick a language and make it current. First source wins: flag, env, file."""
    global _LANG
    env = os.environ if environ is None else environ
    # Read workspace selection without consuming its actual CLI arguments.
    selected_workspace = workspace
    if selected_workspace is None and argv:
        for i, item in enumerate(argv[1:], 1):
            if item == "--":
                break
            if item in ("--workspace", "-w") and i + 1 < len(argv):
                selected_workspace = Path(argv[i + 1]).expanduser()
            elif item.startswith("--workspace="):
                selected_workspace = Path(item.split("=", 1)[1]).expanduser()
    if selected_workspace is None and env.get("LOBSTER_WORKSPACE"):
        selected_workspace = Path(env["LOBSTER_WORKSPACE"]).expanduser()
    chosen = consume_lang_flag(argv)
    if not chosen:
        chosen = env.get("LOBSTER_LANG")
    if not chosen:
        chosen = read_workspace_lang(selected_workspace)
    code = canonicalize(chosen)
    if not known_overlay(code):
        code = "en"
    _LANG = code
    return _LANG


def set_lang(lang: Optional[str]) -> str:
    """Set the process language. Unknown codes fall back to English."""
    global _LANG
    code = canonicalize(lang)
    if not known_overlay(code):
        code = "en"
    _LANG = code
    return _LANG


@contextmanager
def use_lang(lang: Optional[str]) -> Iterator[str]:
    """Temporarily switch the process language, then restore it.

    Machine-readable output (JSON) renders under ``use_lang("en")`` so keys,
    labels and enum-like values never depend on the UI language.
    """
    global _LANG
    previous = _LANG
    try:
        yield set_lang(lang)
    finally:
        _LANG = previous


def write_workspace_lang(workspace: Path, lang: str) -> str:
    """Store ``lang`` in provider_config.json without dropping other keys."""
    code = canonicalize(lang)
    if not known_overlay(code):
        code = "en"
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "provider_config.json"
    data: dict = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("provider_config.json must contain a JSON object")
        data = loaded
    data["lang"] = code
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=workspace,
            prefix=".provider-config-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(data, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return set_lang(code)


_PHRASES: Optional[dict[str, str]] = None


def _phrases() -> dict[str, str]:
    global _PHRASES
    if _PHRASES is None:
        path = _DIR / "phrases.zh.json"
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            _PHRASES = loaded if isinstance(loaded, dict) else {}
        else:
            _PHRASES = {}
    return _PHRASES


def h(english: str) -> str:
    """Translate one English UI sentence. Unknown sentences stay English."""
    if _LANG != "zh":
        return english
    return _phrases().get(english, english)


def reply_language_instruction() -> str:
    """One supervisor sentence. Empty unless the UI language is Chinese."""
    if _LANG != "zh":
        return ""
    return (
        "Reply to the user in easy Simplified Chinese. "
        "Keep tool names, file paths, gene symbols, accessions, and code in English."
    )


def t(key: str, **variables: object) -> str:
    """Return the active translation, or English, or the key if both lack it."""
    text = _load(_LANG).get(key)
    if text is None:
        text = _load("en").get(key, key)
    if variables:
        try:
            return text.format(**variables)
        except (IndexError, KeyError, ValueError):
            return text
    return text
