"""What to tell a user, or a host, that has no provider setup yet.

Engine free: only the wizard manifest (names of the credential variables) and the
provider names. Values of environment variables are never read, only whether a
variable is set. ``lobster init`` and ``lobster_overview`` print the same lines.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

# The code a refusal starts with ("<code>: <text>"), for hosts and tests to match on.
NO_SETUP_CODE = "no_provider_setup"


def key_variables(provider: str) -> List[str]:
    """The environment variables that hold *provider*'s credentials."""
    from lobster.ui.wizard.manifest import _PROVIDER_CREDENTIALS

    return [f.env_var or f.key for f in _PROVIDER_CREDENTIALS.get(provider, [])]


def providers_with_keys(env: Optional[Mapping[str, str]] = None) -> List[str]:
    """Providers whose key variables are all set in *env* (default: the shell).

    A provider that needs no key (Ollama) is never listed: a key variable is the
    only evidence of a choice, and an empty one is none.
    """
    from lobster.config.constants import VALID_PROVIDERS

    env = os.environ if env is None else env
    found = []
    for provider in VALID_PROVIDERS:
        names = key_variables(provider)
        if names and all(env.get(name) for name in names):
            found.append(provider)
    return found


def variables_found(
    provider: str, env: Optional[Mapping[str, str]] = None
) -> List[str]:
    """Names (never values) of *provider*'s key variables that are set."""
    env = os.environ if env is None else env
    return [name for name in key_variables(provider) if env.get(name)]


def lobster_path() -> str:
    """The lobster executable of the install serving this process: the script that
    started it, else the one beside its Python; "lobster" when neither is found.
    A host shell's PATH can hold another version, so printed commands carry this."""
    names = ("lobster", "lobster.exe")
    started = [Path(sys.argv[0])] if sys.argv and sys.argv[0] else []
    beside = [Path(sys.executable).parent / name for name in names]
    for candidate in started + beside:
        if (
            candidate.name in names
            and candidate.is_file()
            and os.access(candidate, os.X_OK)
        ):
            return os.path.abspath(candidate)
    return "lobster"


class Placeholder(str):
    """A word the reader fills in (``<folder>``); the only value printed unquoted."""


def quote(arg: str) -> str:
    if isinstance(arg, Placeholder):
        return str(arg)
    return subprocess.list2cmdline([arg]) if os.name == "nt" else shlex.quote(arg)


def command_line(*args: str, command: Optional[str] = None) -> str:
    """A command line for *args*; *command* is the executable (default: this install's)."""
    return " ".join([command or quote(lobster_path()), *(quote(a) for a in args)])


def suggested_lines(
    folder: str,
    env: Optional[Mapping[str, str]] = None,
    command: Optional[str] = None,
    extra: Sequence[str] = (),
) -> List[str]:
    """One complete command per provider whose key is in the environment, else the
    terminal form. *extra* (the caller's own flags, e.g. ``["--model", "m"]``) is
    repeated on each line. Names only: no value is read into the text."""
    found = providers_with_keys(env)
    if found:
        return [
            command_line(
                "init", folder, "--provider", provider, *extra, command=command
            )
            for provider in found
        ]
    return [
        f"In a terminal: {command_line('init', folder, command=command)}",
        f"Options: {command_line('init', '--help', command=command)}",
    ]


def no_setup_message(
    folder: str,
    env: Optional[Mapping[str, str]] = None,
    command: Optional[str] = None,
    extra: Sequence[str] = (),
) -> str:
    """The refusal for a project made without a terminal when no provider is set."""
    found = providers_with_keys(env)
    lines = [
        f"{NO_SETUP_CODE}: no provider is set up, so nothing was changed "
        "(no project provider file, no global provider file, no LOBSTER_LLM_PROVIDER)."
    ]
    for provider in found:
        names = ", ".join(variables_found(provider, env))
        lines.append(f"{names} is set in the environment for {provider}.")
    lines.append("Run one of:" if found else "Run:")
    lines.extend(f"  {line}" for line in suggested_lines(folder, env, command, extra))
    return "\n".join(lines)
