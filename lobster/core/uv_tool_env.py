"""Detection and command building for uv tool environments.

When Lobster is installed via `uv tool install lobster-ai`, the virtualenv
contains a `uv-receipt.toml` at its prefix. This module detects that receipt
and builds `uv tool install` commands that preserve existing --with packages.
"""

import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NoReturn, Optional

logger = logging.getLogger(__name__)


@dataclass
class UvToolEnvInfo:
    """Information about the current uv tool environment."""

    tool_name: str
    tool_extras: list[str] = field(default_factory=list)
    installed_packages: list[str] = field(default_factory=list)
    prefix: Path = field(default_factory=lambda: Path(sys.prefix))
    # Receipt entries by package name, as uv wrote them (new receipt format only).
    entries: dict[str, dict] = field(default_factory=dict)


def _requirement(entry: dict) -> str:
    """Render a receipt entry back into the requirement uv installed."""
    text = entry["name"]
    if entry.get("extras"):
        text += f"[{','.join(entry['extras'])}]"
    # shortcut: git and editable entries fall back to the bare name.
    if entry.get("path") or entry.get("directory"):
        return f"{text} @ {Path(entry.get('path') or entry['directory']).as_uri()}"
    if entry.get("url"):
        return f"{text} @ {entry['url']}"
    return text + entry.get("specifier", "")


def lobster_release() -> tuple[int, ...]:
    """This Lobster's release numbers, without a local build tag (1.3.0)."""
    from packaging.version import Version

    import lobster.version

    return Version(lobster.version.__version__).release


def detect_uv_tool_env() -> UvToolEnvInfo | None:
    """Detect if running inside a uv tool environment.

    Checks for ``uv-receipt.toml`` at ``sys.prefix``, parses it with
    ``tomllib``, and returns environment info.

    Returns:
        UvToolEnvInfo if inside a uv tool env, None otherwise.
    """
    receipt_path = Path(sys.prefix) / "uv-receipt.toml"
    if not receipt_path.exists():
        return None

    try:
        import tomllib
    except ModuleNotFoundError:
        # Python < 3.11 fallback (shouldn't happen for lobster 3.12+)
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return None

    try:
        with open(receipt_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return None

    tool_section = data.get("tool", {})
    tool_name = tool_section.get("name", "lobster-ai")

    # Extract installed package names from requirements list
    # Format varies by uv version:
    #   Old: ["lobster-ai==1.0.0", "lobster-transcriptomics>=0.1"]
    #   New: [{ name = "lobster-ai", extras = ["full"] }]
    requirements = tool_section.get("requirements", [])
    packages: list[str] = []
    tool_extras: set[str] = set()
    entries: dict[str, dict] = {}
    for req in requirements:
        if isinstance(req, dict):
            # New uv format: {name: "pkg", extras: [...], ...}
            name = req.get("name", "")
            if name:
                entries[name] = req
            if name == tool_name:
                tool_extras.update(req.get("extras", []) or [])
        else:
            # Old uv format: "pkg==1.0.0" or "pkg>=0.1"
            raw_req = req
            name = raw_req
            if raw_req.startswith(f"{tool_name}[") and "]" in raw_req:
                extras_part = raw_req.split("[", 1)[1].split("]", 1)[0]
                for extra in extras_part.split(","):
                    extra = extra.strip()
                    if extra:
                        tool_extras.add(extra)
            for sep in (">=", "<=", "==", "!=", "~=", ">", "<", "["):
                name = name.split(sep)[0]
        name = name.strip()
        if name:
            packages.append(name)

    return UvToolEnvInfo(
        tool_name=tool_name,
        tool_extras=sorted(tool_extras),
        installed_packages=packages,
        prefix=Path(sys.prefix),
        entries=entries,
    )


def is_uv_tool_env() -> bool:
    """Check if running inside a uv tool environment."""
    return detect_uv_tool_env() is not None


def build_tool_install_command(
    extras: list[str] | None = None,
    with_packages: list[str] | None = None,
    *,
    fresh: bool = False,
) -> list[str]:
    """Build a ``uv tool install`` command preserving existing packages.

    Reads the current receipt to keep all existing ``--with`` packages,
    then appends any new packages from *with_packages*.

    Args:
        extras: Extras for the main package (e.g. ``["full", "anthropic"]``).
        with_packages: Additional packages to include via ``--with``.
        fresh: Take lobster-ai from the index (an upgrade) instead of keeping
            how it was installed.

    Returns:
        Command as a list of strings suitable for ``subprocess.run()``.
    """
    uv_path = shutil.which("uv") or "uv"

    info = detect_uv_tool_env()

    merged_extras: set[str] = set(info.tool_extras if info else [])
    if extras:
        merged_extras.update(extras)

    # Keep how Lobster itself was installed (a wheel stays that wheel). A bare
    # name gets a floor at this version: never `==`, uv would then refuse to
    # upgrade the tool.
    own = {} if fresh or not info else dict(info.entries.get("lobster-ai", {}))
    own.update(name="lobster-ai", extras=sorted(merged_extras))
    base_pkg = _requirement(own)
    keeps_source = any(
        own.get(key) for key in ("specifier", "path", "directory", "url")
    )
    if not fresh and not keeps_source:
        base_pkg += ">=" + ".".join(map(str, lobster_release()))

    cmd = [uv_path, "tool", "install", base_pkg]

    # Collect existing --with packages from receipt (excluding the main tool)
    existing_with: set[str] = set()
    if info:
        for pkg in info.installed_packages:
            # Normalize for comparison
            norm = pkg.lower().replace("-", "_")
            if norm not in ("lobster_ai", "lobster"):
                entry = info.entries.get(pkg)
                existing_with.add(_requirement(entry) if entry else pkg)

    # Add new packages
    if with_packages:
        for pkg in with_packages:
            existing_with.add(pkg)

    # Append --with flags
    for pkg in sorted(existing_with):
        cmd.extend(["--with", pkg])

    # Stay on this Python: unpinned, uv picks its newest one, which Lobster may
    # not support. A version, not sys.executable: --reinstall deletes this venv.
    cmd.extend(["--python", f"{sys.version_info[0]}.{sys.version_info[1]}"])
    return cmd


def _windows_join(argv: list[str]) -> str:
    # Double quotes work in PowerShell and cmd; ">=" and "[" bare do not, and
    # single quotes are literal in cmd.
    return " ".join(
        f'"{arg}"' if any(c in arg for c in " []<>=@&|^") else arg for arg in argv
    )


def command_line(argv: list[str]) -> str:
    """A command the user can paste into the shell they are likely in.

    shortcut: on Windows a quoted path first (e.g. a python.exe under a folder
    with a space) is an expression in PowerShell; callers print plain ``uv``.
    """
    import shlex

    return _windows_join(argv) if sys.platform == "win32" else shlex.join(argv)


def write_pending_install(argv: list[str]) -> None:
    """Leave the printed reinstall for the Windows installer (best effort)."""
    import json
    import os
    from datetime import datetime, timezone

    from lobster.config.global_config import pending_install_path
    from lobster.version import __version__

    path = pending_install_path()
    record = {
        "version": 1,
        "argv": argv,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lobster": __version__,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        partial.write_text(json.dumps(record), encoding="utf-8")
        os.replace(partial, path)
    except OSError as exc:  # the printed command still works by hand
        logger.debug(f"Could not write {path}: {exc}")


def clear_pending_install() -> None:
    """Remove a reinstall an earlier init left for the installer."""
    from lobster.config.global_config import pending_install_path

    try:
        pending_install_path().unlink(missing_ok=True)
    except OSError as exc:
        logger.debug(f"Could not remove the pending install: {exc}")


def windows_locks_running_tool() -> bool:
    """Windows locks the running lobster.exe and its venv; uv cannot replace them."""
    return sys.platform == "win32"


def run_tool_install(
    cmd: list[str],
    next_step: Optional[str] = None,
    after: Optional[Callable[[bool], None]] = None,
) -> NoReturn:
    """Run a ``uv tool install`` that replaces this tool's venv, then exit.

    After ``uv tool install`` completes, the venv on disk is replaced while
    this process still has OLD modules loaded in memory. Rich lazy-loads
    _unicode_data modules on demand, so any console.print() after the swap
    crashes with ModuleNotFoundError. Rules after subprocess.run():

    1. NO Rich (console.print, Panel, Prompt, etc.)
    2. Plain print() only, respecting NO_COLOR
    3. os._exit() to skip Python teardown (atexit, __del__, etc.)
       Callers save their config before calling this, or pass ``after``:
       called with whether the install worked, after the swap, so it may
       use only builtins and ``os``.
    """
    import os
    import shlex
    import subprocess  # nosec B404 # re-runs the installer with an argv list

    from lobster.i18n import h, t

    if windows_locks_running_tool():
        # A reinstall from inside could fail halfway and leave no lobster.
        # Plain "uv": PowerShell reads a leading quoted path as an expression.
        line = _windows_join(["uv", *cmd[1:]])
        print("  " + t("cli.init.windows_tool_install", command=line))
        sys.stdout.flush()
        os._exit(0)

    # Translate everything before the swap.
    ok = h("✓ Packages installed successfully.")
    next_step = "  " + (next_step or t("cli.init.next_chat"))
    failed = h("✗ Installation failed.")
    manual = h("  Run manually: {cmd_str}").format(cmd_str=shlex.join(cmd))

    # Stream output so the user sees resolver/download progress
    # instead of staring at a frozen terminal.
    result = subprocess.run(cmd)  # nosec B603 # argv shown before; no shell

    # === DANGER ZONE: venv on disk has been replaced ===
    if after is not None:
        after(result.returncode == 0)
    color = os.environ.get("NO_COLOR") is None and sys.stdout.isatty()
    if result.returncode == 0:
        print(f"  \033[32m{ok}\033[0m" if color else f"  {ok}")
        print(next_step)
    else:
        print(f"  \033[31m{failed}\033[0m" if color else f"  {failed}")
        print(manual)
    sys.stdout.flush()
    os._exit(result.returncode)
