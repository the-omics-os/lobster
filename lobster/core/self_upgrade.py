"""Self-upgrade support for the ``lobster`` CLI.

Builds the correct upgrade command for the detected install method:

* ``uv tool`` installs (receipt at ``sys.prefix``): rebuild the exact
  ``uv tool install --upgrade`` command from the receipt so extras and
  ``--with`` packages are preserved and refreshed. Plain ``uv tool
  install`` is a no-op on an already-installed tool, so ``--upgrade`` is
  required (verified against uv 0.10).
* Virtualenv installs: ``uv pip install --upgrade lobster-ai`` targeting
  the running interpreter, falling back to ``python -m pip install
  --upgrade lobster-ai`` when uv is not on PATH.
* Anything else (system Python, conda, ...): never guess — print manual
  guidance instead.

The upgrade runs in a subprocess. The running process keeps executing
already-loaded code, so users must restart ``lobster`` to pick up the
new version (``run_upgrade`` prints this).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from lobster.i18n import h

if TYPE_CHECKING:
    from rich.console import Console

from dataclasses import dataclass

# Upgrade guidance shown when the install method cannot be detected.
_MANUAL_COMMAND = "uv tool install --upgrade 'lobster-ai[full]'  # or: uv pip install --upgrade lobster-ai"


@dataclass
class UpgradePlan:
    """A resolved upgrade strategy.

    Attributes:
        method: How lobster-ai was installed / how to upgrade
            (``uv-tool``, ``uv-pip``, ``pip``, or ``manual``).
        command: Argument vector to execute (empty for manual plans).
        display: Human-readable form of the command (or guidance).
        reason: Explanation for unsupported (manual) plans.
        supported: False when the plan only prints guidance.
    """

    method: str
    command: list
    display: str
    reason: str = ""
    supported: bool = True


def _manual_plan(reason: str, suggestion: Optional[str] = None) -> UpgradePlan:
    return UpgradePlan(
        method="manual",
        command=[],
        display=suggestion or _MANUAL_COMMAND,
        reason=reason,
        supported=False,
    )


def _is_virtualenv() -> bool:
    """True when the running interpreter lives in a virtualenv."""
    return (Path(sys.prefix) / "pyvenv.cfg").exists()


def is_editable_install() -> bool:
    """True when lobster-ai is an editable (development) install.

    Reads the distribution's PEP 610 ``direct_url.json`` and checks
    ``dir_info.editable``. Any metadata failure means "not editable".
    """
    import json
    from importlib import metadata

    try:
        raw = metadata.distribution("lobster-ai").read_text("direct_url.json")
        if not raw:
            return False
        return bool(json.loads(raw).get("dir_info", {}).get("editable") is True)
    except Exception:
        return False


_EDITABLE_NOTE = (
    "This is an editable development install; "
    "update it with git pull and reinstall instead."
)


def detect_upgrade_plan() -> UpgradePlan:
    """Detect the install method and build the matching upgrade command."""
    from lobster.core.uv_tool_env import detect_uv_tool_env

    uv_path = shutil.which("uv")

    tool_env = detect_uv_tool_env()
    if tool_env is not None:
        from lobster.core.uv_tool_env import build_tool_install_command

        base_pkg = "lobster-ai"
        cmd = list(build_tool_install_command())
        if len(cmd) >= 4 and cmd[1:3] == ["tool", "install"]:
            base_pkg = cmd[3]
        if uv_path is None:
            return _manual_plan(
                "Running from a uv tool environment but `uv` is not on PATH.",
                suggestion=f"uv tool install --upgrade {base_pkg}",
            )
        # Plain `uv tool install` is a no-op on installed tools; --upgrade
        # refreshes the tool and its --with packages in place.
        if len(cmd) >= 4 and cmd[1:3] == ["tool", "install"]:
            cmd.insert(3, "--upgrade")
            return UpgradePlan(method="uv-tool", command=cmd, display=" ".join(cmd))
        return _manual_plan(
            "Could not build a uv tool upgrade command from the install receipt."
        )

    if _is_virtualenv():
        if uv_path is not None:
            cmd = [
                uv_path,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--upgrade",
                "lobster-ai",
            ]
            return UpgradePlan(method="uv-pip", command=cmd, display=" ".join(cmd))
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "lobster-ai"]
        return UpgradePlan(method="pip", command=cmd, display=" ".join(cmd))

    return _manual_plan(
        "Could not determine how lobster-ai was installed "
        "(no uv tool receipt, no virtualenv)."
    )


def run_upgrade(console: "Console", dry_run: bool = False) -> int:
    """Execute (or print) the resolved upgrade plan.

    Args:
        console: Rich console for user-facing output.
        dry_run: Print the command instead of running it.

    Returns:
        Process exit code (0 on success or dry-run, 1 on unsupported
        environments, the subprocess code when the upgrade fails).
    """
    from lobster.config.version_check import clear_update_cache

    plan = detect_upgrade_plan()
    editable = is_editable_install()

    if editable:
        if dry_run:
            console.print(f"[bold white]{h('Dry run — would run:')}[/bold white]")
            console.print(f"  [cyan]{plan.display}[/cyan]")
        console.print(f"[yellow]⚠[/yellow] {_EDITABLE_NOTE}")
        return 0 if dry_run else 1

    if not plan.supported:
        console.print(f"[yellow]⚠[/yellow] {plan.reason}")
        console.print(
            f"  Update manually with: [bold white]{plan.display}[/bold white]"
        )
        return 1

    if dry_run:
        console.print(f"[bold white]{h('Dry run — would run:')}[/bold white]")
        console.print(f"  [cyan]{plan.display}[/cyan]")
        return 0

    console.print(f"[dim]{h('Running:')} {plan.display}[/dim]")
    result = subprocess.run(plan.command)

    if result.returncode != 0:
        console.print(f"[red]✗[/red] Upgrade failed (exit code {result.returncode}).")
        console.print("  You can retry `lobster upgrade` or update manually:")
        console.print(f"  [bold white]{plan.display}[/bold white]")
        return result.returncode or 1

    # Best-effort: drop the cached latest-version so the next launch
    # re-checks PyPI instead of showing a stale update notification.
    clear_update_cache()
    console.print(f"[green]✓[/green] {h('lobster-ai upgraded.')}")
    console.print(f"[dim]{h('Restart `lobster` to use the new version.')}[/dim]")
    return 0
